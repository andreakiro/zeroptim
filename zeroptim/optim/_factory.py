from typing import Dict, Any

import torch.nn as nn
import torch.optim as optim

from zeroptim.configs._types import Config
from zeroptim.supported import __supported_optims__, __supported_criterions__


class OptimFactory:
    @staticmethod
    def init_optimizer(config: Config, model: nn.Module) -> optim.Optimizer:
        optimizer_type: str = config.optim.optimizer_type
        if optimizer_type not in __supported_optims__:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        opt_params: Dict[str, Any] = config.optim.opt_params

        if optimizer_type in ["mezo", "smartes"]:
            # return a zero-order optimizer
            epsilon = config.optim.epsilon
            sub_opt_type = config.optim.sub_optimizer_type
            sub_optimizer = __supported_optims__[sub_opt_type](
                model.parameters(), **opt_params
            )

            if optimizer_type == "mezo":
                return __supported_optims__[optimizer_type](sub_optimizer, epsilon)
            if optimizer_type == "smartes":
                return __supported_optims__[optimizer_type](
                    sub_optimizer, model, epsilon
                )

        return __supported_optims__[optimizer_type](model.parameters(), **opt_params)

    @staticmethod
    def init_criterion(config: Config) -> nn.Module:
        criterion_type: str = config.optim.criterion_type
        if criterion_type not in __supported_criterions__:
            raise ValueError(f"Unsupported criterion type: {criterion_type}")
        return __supported_criterions__[criterion_type]()
