from typing import Optional, Tuple, List, Dict, ClassVar, Any
from pathlib import Path
from tqdm import tqdm
import wandb, json

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


class ZeroptimTrainer:

    # Assuming we run from root of project directory
    OUTPUT_DIR: ClassVar[Path] = Path.cwd() / "outputs"

    def __init__(
        self, 
        model: nn.Module,
        dataloader: D.DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        run_config: Config
    ) -> None:
        
        # initialize trainer
        self.model = model
        self.loader = dataloader
        self.opt = optimizer
        self.crit = criterion
        self.config = run_config


    @staticmethod
    def from_config(config: Config) -> "ZeroptimTrainer":
        model = ModelFactory.init_model(config)
        dataloader = DataLoaderFactory.init_loader(config)
        optimizer = OptimFactory.init_optimizer(config, model)
        criterion = OptimFactory.init_criterion(config)
        return ZeroptimTrainer(model, dataloader, optimizer, criterion, config)


    def train(
        self,
        epochs: int = 100,
        max_iters: Optional[int | None] = None,
        val_loader: Optional[D.DataLoader] = None,
    ) -> Tuple[List[float], List[float], List[float]]:
        
        self.activate_wnb()

         # initialize output directory
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

        def closure(inputs, targets, with_backward=False):
            # optimization-step closure
            self.opt.zero_grad()
            loss = self.crit(self.model(inputs), targets)
            if with_backward:
                loss.backward()
            return loss

        # initialize metrics
        n_epochs, n_iters = 0, 0
        n_iters_tot = len(self.loader) * epochs
        train_losses, val_losses = [], []
        train_losses_per_iter = []
        jvps, vhvs = [], []

        self.model.train()
        for epoch_idx in (pbar := tqdm(range(epochs))):
            if max_iters is not None and n_iters >= max_iters: break
            epoch_loss = 0

            for batch_idx, (inputs, targets) in enumerate(self.loader):
                if max_iters is not None and n_iters >= max_iters: break
                
                device = self.config.env.device
                inputs, targets = inputs.to(device), targets.to(device)
                assert len(inputs.shape) >= 2, "Need a batch size dimension!"
                inputs = inputs.view(inputs.shape[0], -1)  # flatten for MLP

                prev_params = tuple([p.clone() for p in self.model.parameters()])

                # take optimization step
                loss = self.opt.step(
                    lambda: closure(
                        inputs,
                        targets,
                        with_backward=not isinstance(self.opt, (MeZO, SmartES)),
                    )
                )

                curr_params = tuple([p.clone() for p in self.model.parameters()])
                vs = tuple(
                    [
                        p2.detach() - p1.detach()
                        for p2, p1 in zip(curr_params, prev_params)
                    ]
                )

                # compute jvp and hvp
                tmp_params, names = utils.make_functional(self.model)
                _, jvp = torch.autograd.functional.jvp(func_fwd, prev_params, vs)
                _, hvp = torch.autograd.functional.hvp(func_fwd, prev_params, vs)
                vhv = sum((v * hv).sum() for v, hv in zip(vs, hvp))
                utils.restore_functional(self.model, tmp_params, names)

                # append metrics
                train_losses_per_iter.append(loss.item())
                jvps.append(jvp.item())
                vhvs.append(vhv.item())
                epoch_loss += loss.item()

                wandb.log({
                    "train_loss": train_losses_per_iter[-1],
                    "jvp": jvps[-1],
                    "vhv": vhvs[-1],
                })

                # update tqdm progress bar
                pbar.set_description(
                    f"epoch {epoch_idx+1}/{epochs} iter {n_iters+1}/{n_iters_tot} | train loss {loss.item():.3f}"
                )
                pbar.update(1)
                n_iters += 1

            # append average train loss for current epoch
            train_losses.append(epoch_loss / len(self.loader))

            # evaluate on validation set
            if val_loader is not None:
                val_loss = self.test(val_loader)
                val_losses.append(val_loss)

            n_epochs += 1

        pbar.close()

        res = {
            "model": self.model.__class__.__name__.lower(),
            "n_epochs": n_epochs,
            "n_iters": n_iters,
            "train_losses": train_losses,
            "train_losses_per_iter": train_losses_per_iter,
            **({"val_losses": val_losses} if val_losses else {}),
            "jvps": jvps,
            "vhvs": vhvs,
        }

        # save results to disk
        with open(self.BASE / "results.json", "w") as f:
            json.dump(res, f, indent=4)

        # save final checkpoint to disk
        pth = Path(self.BASE / "checkpoints")
        pth.mkdir(parents=True, exist_ok=True)
        filename = f"epoch_{res['n_epochs']-1}.pt"
        torch.save(self.model.state_dict(), str(pth / filename))

        return res


    def test(
        self,
        test_loader: D.DataLoader
    ) -> float:
        
        self.model.eval()

        # initialize metrics
        total_loss, total_samples = 0, 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                device = self.config.env.device
                inputs, targets = inputs.to(device), targets.to(device)
                assert len(inputs.shape) >= 2, "Need a batch size dimension!"
                inputs = inputs.view(inputs.shape[0], -1)  # flatten for MLP
                loss = self.crit(self.model(inputs), targets)
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

        avg_loss = total_loss / total_samples

        return avg_loss


    def activate_wnb(self) -> None:
        wandb.init(
            project=self.config.wandb.project_name,
            name=self.config.wandb.run_name,
            mode=self.config.wandb.mode,
            config=self.config.model_dump()
        )

        if self.config.wandb.mode == "online":
            self.config.wandb.run_name = wandb.run.name
            