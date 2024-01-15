from typing import Optional, Tuple, List, ClassVar, Literal
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

from loguru import logger


class ZeroptimTrainer:
    # Assuming we run from root of project directory
    OUTPUT_DIR: ClassVar[Path] = Path.cwd() / "outputs"
    LAYERWISE: bool = False
    OVER_LANDSCAPE: Literal["batch", "partial", "global"]
    NUM_BATCHES: float = 10  # for partial landscape only
    DIRECTED: bool

    def __init__(
        self,
        model: nn.Module,
        dataloader: D.DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        run_config: Config,
        directed: bool = False,
        over_landscape: str = "batch",
    ) -> None:
        # initialize trainer
        self.model = model
        self.loader = dataloader
        self.opt = optimizer
        self.crit = criterion
        self.config = run_config
        self.DIRECTED = directed
        self.OVER_LANDSCAPE = over_landscape

    @staticmethod
    def from_config(config: Config) -> "ZeroptimTrainer":
        model = ModelFactory.init_model(config)
        dataloader = DataLoaderFactory.get_loader(config)
        optimizer = OptimFactory.get_optimizer(config, model)
        criterion = OptimFactory.init_criterion(config)
        return ZeroptimTrainer(model, dataloader, optimizer, criterion, config)

    def compute_accuracy(self, outputs, targets):
        _, predicted = torch.max(outputs.data, 1)
        total = targets.size(0)
        correct = (predicted == targets).sum().item()
        accuracy = 100 * correct / total
        return accuracy

    def train(
        self,
        epochs: int = 100,
        max_iters: Optional[int | None] = None,
        val_loader: Optional[D.DataLoader] = None,
    ) -> Tuple[List[float], List[float], List[float]]:
        self.activate_wnb()

        # initialize output directory
        rn = self.config.wandb.run_name
        ts = self.config.wandb.timestamp
        self.BASE = Path(self.OUTPUT_DIR / "-".join((ts, rn)))
        self.BASE.mkdir(parents=True, exist_ok=True)
        save(self.config, str(self.BASE / "config.yaml"))

        print(f"Starting experiment {self.config.wandb.run_name}")
        print(f"Saving outputs and results to {self.BASE}")

        def func_fwd(*model_params):
            for name, p in zip(names, model_params):
                utils.set_attr(self.model, name.split("."), p)
            return self.crit(self.model(inputs), targets)

        def closure(outputs, targets, with_backward=False):
            # optimization-step closure
            self.opt.zero_grad()
            loss = self.crit(outputs, targets)
            if with_backward:
                loss.backward()
            return loss

        iters_per_epoch = len(self.loader)

        # initialize metrics
        n_epochs, n_iters = 0, 0
        n_iters_tot = iters_per_epoch * epochs
        train_loss, val_loss, train_acc = [], [], []
        train_loss_per_iter, train_acc_per_iter = [], []
        jvps_per_iter, vhvs_per_iter = [], []

        self.model.train()
        for epoch_idx in (pbar := tqdm(range(epochs))):
            if max_iters is not None and n_iters >= max_iters:
                break
            epoch_loss = 0

            for batch_idx, (inputs, targets) in enumerate(self.loader):
                if max_iters is not None and n_iters >= max_iters:
                    break

                device = self.config.env.device
                inputs, targets = inputs.to(device), targets.to(device)
                assert len(inputs.shape) >= 2, "Need a batch size dimension!"
                inputs = inputs.view(inputs.shape[0], -1)  # flatten for MLP
                outputs = self.model(inputs)

                prev_params = tuple([p.clone() for p in self.model.parameters()])

                # take optimization step
                loss = self.opt.step(
                    lambda: closure(
                        outputs,
                        targets,
                        with_backward=not isinstance(self.opt, (MeZO, SmartES)),
                    )
                )

                curr_params = tuple([p.clone() for p in self.model.parameters()])
                vs = [
                    p2.detach() - p1.detach()
                    for p2, p1 in zip(curr_params, prev_params)
                ]

                if self.DIRECTED:
                    named_params = [
                        (name, p.clone().detach())
                        for name, p in self.model.named_parameters()
                    ]

                    # compute jvp and hvp for each layer
                    for idx, (name, weights) in enumerate(named_params):
                        if "weight" in name:
                            U, S, V = torch.linalg.svd(weights.data)
                            u1, s1, v1 = U[:, 0], S[0], V[:, 0]
                            vs[idx] = s1 * torch.outer(u1, v1)

                # compute jvp and hvp for entire model
                tmp_params, names = utils.make_functional(self.model)
                vs = tuple(vs)

                if self.OVER_LANDSCAPE == "batch":
                    _, jvp = torch.autograd.functional.jvp(func_fwd, prev_params, vs)
                    _, hvp = torch.autograd.functional.hvp(func_fwd, prev_params, vs)
                    vhv = sum((v * hv).sum() for v, hv in zip(vs, hvp))
                    jvp, vhv = jvp.item(), vhv.item()

                elif self.OVER_LANDSCAPE == "partial":
                    agg_jvp, agg_vhv, count = 0.0, 0.0, 0.0
                    for inps, trgts in sample(
                        self.loader, num_batches=self.NUM_BATCHES
                    ):
                        sz = inps.size(0)
                        inputs, targets = inps.to(device), trgts.to(device)
                        inputs = inputs.view(inputs.shape[0], -1)
                        _, jvp_ = torch.autograd.functional.jvp(
                            func_fwd, prev_params, vs
                        )
                        _, hvp_ = torch.autograd.functional.hvp(
                            func_fwd, prev_params, vs
                        )
                        vhv_ = sum((v * hv).sum() for v, hv in zip(vs, hvp_))
                        agg_jvp += jvp_.item() * sz
                        agg_vhv += vhv_.item() * sz
                        count += sz

                    jvp = agg_jvp / count
                    vhv = agg_vhv / count

                elif self.OVER_LANDSCAPE == "global":
                    agg_jvp, agg_vhv, count = 0.0, 0.0, 0.0
                    for inps, trgts in self.loader:
                        sz = inps.size(0)
                        inputs, targets = inps.to(device), trgts.to(device)
                        inputs = inputs.view(inputs.shape[0], -1)
                        _, jvp_ = torch.autograd.functional.jvp(
                            func_fwd, prev_params, vs
                        )
                        _, hvp_ = torch.autograd.functional.hvp(
                            func_fwd, prev_params, vs
                        )
                        vhv_ = sum((v * hv).sum() for v, hv in zip(vs, hvp_))
                        agg_jvp += jvp_.item() * sz
                        agg_vhv += vhv_.item() * sz
                        count += sz

                    jvp = agg_jvp / count
                    vhv = agg_vhv / count

                utils.restore_functional(self.model, tmp_params, names)

                dict = {}
                if self.LAYERWISE:
                    named_params = [
                        (name, p.clone().detach())
                        for name, p in self.model.named_parameters()
                    ]

                    # compute jvp and hvp for each layer
                    for idx, (name, weights) in enumerate(named_params):
                        tangent = [torch.zeros_like(v) for v in vs]

                        if "weight" in name and self.DIRECTED:
                            U, S, V = torch.linalg.svd(weights.data)
                            u1, s1, v1 = U[:, 0], S[0], V[:, 0]
                            tangent[idx] = s1 * torch.outer(u1, v1)

                        else:
                            # tangent defined as last step
                            tangent[idx] = vs[idx]

                        tangent = tuple(tangent)

                        tmp_params, names = utils.make_functional(self.model)
                        _, jvp = torch.autograd.functional.jvp(
                            func_fwd, prev_params, tangent
                        )
                        _, hvp = torch.autograd.functional.hvp(
                            func_fwd, prev_params, tangent
                        )
                        vhv = sum((v * hv).sum() for v, hv in zip(vs, hvp))
                        utils.restore_functional(self.model, tmp_params, names)
                        dict["jvp_" + name] = jvp.item()
                        dict["vhv_" + name] = vhv.item()

                # append metrics
                train_loss_per_iter.append(loss.item())
                train_acc_per_iter.append(self.compute_accuracy(outputs, targets))
                jvps_per_iter.append(jvp)
                vhvs_per_iter.append(vhv)
                epoch_loss += loss.item()

                wandb.log(
                    {
                        "iter": n_iters,
                        "train_loss_per_iter": train_loss_per_iter[-1],
                        "train_accuracy_per_iter": train_acc_per_iter[-1],
                        "jvp_global": jvps_per_iter[-1],
                        "vhv_global": vhvs_per_iter[-1],
                        **dict,
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
                    "epoch": epoch_idx,
                    "train_loss_per_epoch": train_loss[-1],
                    "train_accuracy_per_epoch": train_acc[-1],
                }
            )

            # evaluate on validation set
            if val_loader is not None:
                val_loss = self.test(val_loader)
                val_loss.append(val_loss)

            n_epochs += 1

        pbar.close()

        res = {
            "model": self.model.__class__.__name__.lower(),
            "n_epochs": n_epochs,
            "n_iters": n_iters,
            "train_loss_per_epoch": train_loss,
            **({"val_loss_per_epoch": val_loss} if val_loss else {}),
            "train_acc_per_epoch": train_acc,
            "train_loss_per_iter": train_loss_per_iter,
            "train_acc_per_iter": train_acc_per_iter,
            "jvps_per_iter": jvps_per_iter,
            "vhvs_per_iter": vhvs_per_iter,
        }

        # save results to disk
        with open(self.BASE / "results.json", "w") as f:
            json.dump(res, f, indent=4)

        # save final checkpoint to disk
        pth = Path(self.BASE / "checkpoints")
        pth.mkdir(parents=True, exist_ok=True)
        filename = f"epoch_{res['n_epochs']-1}.pt"
        torch.save(self.model.state_dict(), str(pth / filename))

        return res

    def test(self, test_loader: D.DataLoader) -> Tuple[float, float]:
        self.model.eval()

        # initialize metrics
        total_loss, total_samples, correct_predictions = 0, 0, 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                device = self.config.env.device
                inputs, targets = inputs.to(device), targets.to(device)
                assert len(inputs.shape) >= 2, "Need a batch size dimension!"
                inputs = inputs.view(inputs.shape[0], -1)  # flatten for MLP
                outputs = self.model(inputs)
                loss = self.crit(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
                correct_predictions += (outputs == targets).sum().item()

        avg_loss = total_loss / total_samples
        avg_accuracy = 100 * correct_predictions / total_samples

        return avg_loss, avg_accuracy

    def activate_wnb(self) -> None:
        wandb.init(
            project=self.config.wandb.project_name,
            name=self.config.wandb.run_name,
            mode=self.config.wandb.mode,
            config=self.config.model_dump(),
        )

        if self.config.wandb.mode == "online":
            self.config.wandb.run_name = wandb.run.name
