from typing import Optional, Tuple, ClassVar, Dict, Any
from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm
import wandb
import json

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as D
import torch

from zeroptim.configs._types import Config
from zeroptim.dataset._factory import DataLoaderFactory
from zeroptim.models._factory import ModelFactory
from zeroptim.optim._factory import OptimFactory

from zeroptim.configs import save
import zeroptim.callbacks.functional as utils
from zeroptim.optim.mezo import MeZO
from zeroptim.optim.smartes import SmartES
from zeroptim.dataset.utils import sample


class BaseTrainer(ABC):
    OUTPUT_DIR: ClassVar[Path] = Path.cwd() / "outputs"

    def __init__(
        self,
        model: nn.Module,
        dataloader: D.DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        run_config: Config,
    ) -> None:
        # initialize trainer
        self.model = model
        self.loader = dataloader
        self.opt = optimizer
        self.crit = criterion
        self.config = run_config

    @classmethod
    def from_config(cls, config: Config) -> "BaseTrainer":
        model = ModelFactory.get_model(config)
        dataloader = DataLoaderFactory.get_loader(config)
        optimizer = OptimFactory.get_optimizer(config, model)
        criterion = OptimFactory.init_criterion(config)
        return cls(model, dataloader, optimizer, criterion, config)

    def activate_wnb(self) -> None:
        wandb.init(
            project=self.config.wandb.project_name,
            name=self.config.wandb.run_name,
            mode=self.config.wandb.mode,
            config=self.config.model_dump(),
        )

        if self.config.wandb.mode == "online":
            self.config.wandb.run_name = wandb.run.name

    def compute_accuracy(self, outputs, targets):
        _, predicted = torch.max(outputs.data, 1)
        total = targets.size(0)
        correct = (predicted == targets).sum().item()
        accuracy = 100 * correct / total
        return accuracy

    @abstractmethod
    def train_loop(
        self,
        epochs: int = 100,
        max_iters: Optional[int | None] = None,
        val_loader: Optional[D.DataLoader] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError()

    def train(
        self,
        epochs: int = 100,
        max_iters: Optional[int | None] = None,
        val_loader: Optional[D.DataLoader] = None,
    ) -> None:
        self.activate_wnb()

        # initialize output directory
        rn = self.config.wandb.run_name
        ts = self.config.wandb.timestamp
        self.BASE = Path(self.OUTPUT_DIR / "-".join((ts, rn)))
        self.BASE.mkdir(parents=True, exist_ok=True)
        save(self.config, str(self.BASE / "config.yaml"))

        # run training loop
        metrics = self.train_loop(
            epochs=epochs, max_iters=max_iters, val_loader=val_loader
        )

        # save results to disk
        with open(self.BASE / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        # save final checkpoint to disk
        pth = Path(self.BASE / "checkpoints")
        pth.mkdir(parents=True, exist_ok=True)
        filename = f"epoch_{metrics['n_epochs']-1}.pt"
        torch.save(self.model.state_dict(), str(pth / filename))

    def test(self, test_loader: D.DataLoader) -> Tuple[float, float]:
        self.model.eval()

        # initialize metrics
        total_loss, total_samples, correct_predictions = 0.0, 0.0, 0.0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                device = self.config.env.device
                inputs, targets = inputs.to(device), targets.to(device)
                assert len(inputs.shape) >= 2, "Need a batch size dimension!"
                outputs = self.model(inputs)
                loss = self.crit(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
                correct_predictions += (outputs == targets).sum().item()

        avg_loss = total_loss / total_samples
        avg_accuracy = 100 * correct_predictions / total_samples

        return avg_loss, avg_accuracy


class ZeroptimTrainer(BaseTrainer):
    def step_loss(self, outputs, targets):
        def closure(outputs, targets, with_backward=False):
            # optimization-step closure
            self.opt.zero_grad()
            loss = self.crit(outputs, targets)
            if with_backward:
                loss.backward()
            return loss

        prev_params = tuple([p.clone().detach() for p in self.model.parameters()])
        with_back = not isinstance(self.opt, (MeZO, SmartES))
        loss = self.opt.step(lambda: closure(outputs, targets, with_backward=with_back))
        post_params = tuple([p.clone().detach() for p in self.model.parameters()])

        return loss, prev_params, post_params

    def svd_filter(self, tangents, top_k=1):
        for idx, (name, weights) in enumerate(self.model.named_parameters()):
            if "weight" in name:
                U, S, V = torch.linalg.svd(weights.data)
                uK, sK, vK = U[:, :top_k], S[:top_k], V[:, :top_k]
                # tangents[idx] = s1 * torch.outer(u1, v1)
                tangents[idx] = (uK * sK) @ vK.T

        return tangents

    def metrics_in_batch(self, inputs, targets, names, primals, tangents, func_fwd):
        state = {"names": names, "inputs": inputs, "targets": targets}
        _, jvp = torch.autograd.functional.jvp(func_fwd, primals, tangents, state=state)
        _, hvp = torch.autograd.functional.hvp(func_fwd, primals, tangents, state=state)
        vhv = sum((v * hv).sum() for v, hv in zip(tangents, hvp))
        jvp, vhv = jvp.item(), vhv.item()
        return jvp, vhv

    def metrics_in_landscape(self, iterator, names, primals, tangents, func_fwd):
        agg_jvp, agg_vhv, count = 0.0, 0.0, 0.0

        for inputs, targets in iterator:
            sz = inputs.size(0)
            device = self.config.env.device
            inputs, targets = inputs.to(device), targets.to(device)
            state = {"names": names, "inputs": inputs, "targets": targets}
            _, jvp_ = torch.autograd.functional.jvp(
                func_fwd, primals, tangents, state=state
            )
            _, hvp_ = torch.autograd.functional.hvp(
                func_fwd, primals, tangents, state=state
            )
            vhv_ = sum((v * hv).sum() for v, hv in zip(tangents, hvp_))
            agg_jvp += jvp_.item() * sz
            agg_vhv += vhv_.item() * sz
            count += sz

        jvp = agg_jvp / count
        vhv = agg_vhv / count

        return jvp, vhv

    def sharpness_hook(self, inputs, targets, prev_params, tangents):
        def func_fwd(*model_params, **kwargs):
            # state-less functional forward pass (except model)
            names = kwargs.get("state").get("names")
            inputs = kwargs.get("state").get("inputs")
            targets = kwargs.get("state").get("targets")
            for name, p in zip(names, model_params):
                utils.set_attr(self.model, name.split("."), p)
            return self.crit(self.model(inputs), targets)

        # store model to make it functional for jvp and hvp
        tmp_params, names = utils.make_functional(self.model)

        if self.config.exp.landscape == "batch":
            jvp, vhv = self.metrics_in_batch(
                inputs, targets, names, prev_params, tangents, func_fwd
            )
        elif self.config.exp.landscape == "partial":
            iterator = sample(self.loader, num_batches=self.config.exp.n_batch)
            jvp, vhv = self.metrics_in_landscape(
                iterator, names, prev_params, tangents, func_fwd
            )
        elif self.config.exp.landscape == "full":
            iterator = self.loader
            jvp, vhv = self.metrics_in_landscape(
                iterator, names, prev_params, tangents, func_fwd
            )

        # restore the model to original state :)
        utils.restore_functional(self.model, tmp_params, names)

        return jvp, vhv

    def train_loop(
        self,
        epochs: int = 100,
        max_iters: Optional[int | None] = None,
        val_loader: Optional[D.DataLoader] = None,
    ) -> Dict[str, Any]:
        # initialize metrics
        n_epochs, n_iters = 0, 0
        iters_per_epoch = len(self.loader)  # num batches per epoch
        n_iters_tot = iters_per_epoch * epochs
        train_loss, val_loss, train_acc = [], [], []
        train_loss_per_iter, train_acc_per_iter = [], []
        jvps_per_iter, vhvs_per_iter = [], []
        layerwise_metrics_per_iter = []

        pbar = tqdm(range(n_iters_tot))

        self.model.train()
        for epoch_idx in range(epochs):
            epoch_loss = 0

            for batch_idx, (inputs, targets) in enumerate(self.loader):
                if max_iters is not None and n_iters >= max_iters:
                    break

                device = self.config.env.device
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                loss, prev_params, post_params = self.step_loss(outputs, targets)
                accuracy = self.compute_accuracy(outputs, targets)

                direction = [p2 - p1 for p2, p1 in zip(post_params, prev_params)]
                tangents = tuple(
                    self.svd_filter(direction)
                    if self.config.exp.directed
                    else direction
                )

                jvp, vhv = self.sharpness_hook(inputs, targets, prev_params, tangents)

                layerwise_metrics = {}
                if self.config.exp.layerwise:
                    # compute jvp and hvp for each layer
                    for idx, (name, weights) in enumerate(
                        self.model.named_parameters()
                    ):
                        L_tangent = [torch.zeros_like(v) for v in tangents]
                        L_tangent[idx] = tangents[idx]
                        L_tangent = tuple(L_tangent)
                        L_jvp, L_vhv = self.sharpness_hook(
                            inputs, targets, prev_params, L_tangent
                        )
                        layerwise_metrics["jvp_per_iter_" + name] = L_jvp
                        layerwise_metrics["vhv_per_iter_" + name] = L_vhv
                    layerwise_metrics_per_iter.append(layerwise_metrics)

                # append metrics
                train_loss_per_iter.append(loss.item())
                train_acc_per_iter.append(accuracy)
                jvps_per_iter.append(jvp)
                vhvs_per_iter.append(vhv)
                epoch_loss += loss.item()

                wandb.log(
                    {
                        "n_iter": n_iters,
                        "train_loss_per_iter": train_loss_per_iter[-1],
                        "train_acc_per_iter": train_acc_per_iter[-1],
                        "jvp_per_iter": jvps_per_iter[-1],
                        "vhv_per_iter": vhvs_per_iter[-1],
                        **layerwise_metrics,
                    }
                )

                # update tqdm progress bar
                pbar.set_description(
                    f"epoch {epoch_idx+1}/{epochs} iter {n_iters+1}/{n_iters_tot} | train loss {loss.item():.3f}"
                )
                pbar.update(1)
                n_iters += 1

            # append average train loss for current epoch
            train_loss.append(epoch_loss / iters_per_epoch)
            train_acc.append(
                sum(train_acc_per_iter[-iters_per_epoch:]) / iters_per_epoch
            )

            wandb.log(
                {
                    "n_epoch": epoch_idx,
                    "train_loss_per_epoch": train_loss[-1],
                    "train_acc_per_epoch": train_acc[-1],
                }
            )

            # evaluate on validation set
            if val_loader is not None:
                val_loss = self.test(val_loader)
                val_loss.append(val_loss)

            n_epochs += 1

        pbar.close()

        metrics = {
            "model": self.model.__class__.__name__.lower(),
            # per iters metrics
            "n_iters": n_iters,
            "train_loss_per_iter": train_loss_per_iter,
            "train_acc_per_iter": train_acc_per_iter,
            "jvps_per_iter": jvps_per_iter,
            "vhvs_per_iter": vhvs_per_iter,
            "layerwise_metrics_per_iter": layerwise_metrics_per_iter,
            # per epoch metrics
            "n_epochs": n_epochs,
            "train_loss_per_epoch": train_loss,
            "train_acc_per_epoch": train_acc,
            **({"val_loss_per_epoch": val_loss} if val_loss else {}),
        }

        return metrics
