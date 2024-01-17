from typing import Optional, Tuple, ClassVar, Dict, Any
from abc import ABC, abstractmethod
from itertools import chain
from pathlib import Path
from tqdm import tqdm
import wandb
import json

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as D
import torch

import zeroptim.utils.plots as plots
from matplotlib import pyplot as plt

from zeroptim.configs._types import Config
from zeroptim.dataset._factory import DataLoaderFactory
from zeroptim.models._factory import ModelFactory
from zeroptim.optim._factory import OptimFactory

from zeroptim.configs import save
import zeroptim.callbacks.functional as utils
from zeroptim.optim.mezo import MeZO
from zeroptim.optim.smartes import SmartES
from zeroptim.dataset.utils import sample
from loguru import logger


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

        # plot results to disk
        parsed = plots.parse_raw_metrics(metrics)
        bigtitle = f"{self.config.optim.optimizer_type} svd={str(self.config.sharpness.svd).lower()} landscape={self.config.sharpness.landscape}"
        fig = plots.scatter_metrics_together(parsed, bigtitle)
        fig.savefig(self.BASE / "scatter.png")

        # save final checkpoint to disk
        pth = Path(self.BASE / "checkpoints")
        pth.mkdir(parents=True, exist_ok=True)
        filename = f"epoch_{metrics['n_epoch']-1}.pt"
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

        prev_params = self.model.named_parameters()
        prev_params = [(n, p.clone().detach()) for (n, p) in prev_params]

        with_back = not isinstance(self.opt, (MeZO, SmartES))
        loss = self.opt.step(lambda: closure(outputs, targets, with_backward=with_back))

        post_params = self.model.named_parameters()
        post_params = [(n, p.clone().detach()) for (n, p) in post_params]

        return loss, prev_params, post_params

    def svd_filter(self, tangents, prev_params, top_k=1):
        for idx, (name, weights) in enumerate(prev_params):
            if "weight" in name:
                U, S, Vh = torch.linalg.svd(weights.data)
                uK, sK, vhK = U[:, :top_k], S[:top_k], Vh[:top_k, :]
                # tangents[idx] = s1 * torch.outer(u1, v1)
                topK_singular_space = (uK * sK) @ vhK
                tangents[idx] = topK_singular_space

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

        if self.config.sharpness.landscape == "batch":
            jvp, vhv = self.metrics_in_batch(
                inputs, targets, names, prev_params, tangents, func_fwd
            )
        elif self.config.sharpness.landscape == "partial":
            iterator = sample(self.loader, num_batches=self.config.sharpness.n_batch)
            iterator = chain([(inputs, targets)], iterator)
            jvp, vhv = self.metrics_in_landscape(
                iterator, names, prev_params, tangents, func_fwd
            )
        elif self.config.sharpness.landscape == "full":
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
        # initialize counters
        n_epochs, n_iters = 0, 0
        iters_per_epoch = len(self.loader)  # num batches per epoch
        n_iters_tot = iters_per_epoch * epochs

        pbar = tqdm(range(n_iters_tot))
        metrics = {"model": self.model.__class__.__name__.lower()}

        self.model.train()
        for epoch_idx in range(epochs):
            epoch_loss, epoch_acc = 0.0, 0.0

            for _, (inputs, targets) in enumerate(self.loader):
                if max_iters is not None and n_iters >= max_iters:
                    break

                device = self.config.env.device
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                loss, prev_params, post_params = self.step_loss(outputs, targets)
                accuracy = self.compute_accuracy(outputs, targets)
                epoch_loss += loss.item()
                epoch_acc += accuracy

                per_iter_metrics = {}
                per_iter_metrics["n_iter"] = n_iters
                per_iter_metrics["train_loss_per_iter"] = loss.item()
                per_iter_metrics["train_acc_per_iter"] = accuracy

                # compute sharpness every other step
                if (
                    self.config.sharpness.frequency >= 1
                    and n_iters % self.config.sharpness.frequency == 0
                ):
                    get_data = lambda named_params: map(lambda x: x[1], named_params)
                    zipper = zip(get_data(post_params), get_data(prev_params))
                    delta_W = [p2 - p1 for p2, p1 in zipper]

                    tangents = tuple(
                        self.svd_filter(delta_W, prev_params)
                        if self.config.sharpness.svd
                        else delta_W
                    )

                    prev_params = tuple(get_data(prev_params))
                    jvp, vhv = self.sharpness_hook(
                        inputs, targets, prev_params, tangents
                    )

                    per_iter_metrics["jvp_per_iter"] = jvp
                    per_iter_metrics["vhv_per_iter"] = vhv

                    if self.config.sharpness.layerwise:
                        # compute jvp and hvp for each layer
                        for idx, (name, _) in enumerate(post_params):
                            L_tangent = [torch.zeros_like(v) for v in tangents]
                            L_tangent[idx] = tangents[idx]
                            L_tangent = tuple(L_tangent)
                            L_jvp, L_vhv = self.sharpness_hook(
                                inputs, targets, prev_params, L_tangent
                            )
                            per_iter_metrics["jvp_per_iter_" + name] = L_jvp
                            per_iter_metrics["vhv_per_iter_" + name] = L_vhv

                # update iters wandb and metrics
                wandb.log(per_iter_metrics)
                metrics.setdefault("per_iter", []).append(per_iter_metrics)

                # move progress bar
                pbar.set_description(
                    f"epoch {epoch_idx+1}/{epochs} iter {n_iters+1}/{n_iters_tot} | train loss {loss.item():.3f}"
                )
                pbar.update(1)
                n_iters += 1

            per_epoch_metrics = {}
            per_epoch_metrics["n_epoch"] = epoch_idx
            per_epoch_metrics["train_loss_per_epoch"] = epoch_loss / iters_per_epoch
            per_epoch_metrics["train_acc_per_epoch"] = epoch_acc / iters_per_epoch

            # evaluate on validation set
            if val_loader is not None:
                val_loss, val_acc = self.test(val_loader)
                per_epoch_metrics["val_loss_per_epoch"] = val_loss
                per_epoch_metrics["val_acc_per_epoch"] = val_acc

            # update epoch wandb and metrics
            wandb.log(per_epoch_metrics)
            metrics.setdefault("per_epoch", []).append(per_epoch_metrics)
            n_epochs += 1

        metrics["n_epoch"] = n_epochs
        metrics["n_iter"] = n_iters

        pbar.close()

        return metrics
