from typing import Any, Dict

import torch.nn as nn

from zeroptim.configs._types import Config
from zeroptim.supported import __supported_models__, __supported_activations__


class ModelFactory:
    @staticmethod
    def get_model(config: Config) -> nn.Module:
        model_type: str = config.model.model_type
        if model_type not in __supported_models__:
            raise ValueError(f"Unsupported model type: {model_type}")
        model_params: Dict[str, Any] = ModelFactory.init_params(
            config.model.model_hparams
        )
        return __supported_models__[model_type](**model_params)

    @staticmethod
    def init_activation(activation_type: str) -> nn.Module:
        if activation_type not in __supported_activations__:
            raise ValueError(f"Unsupported activation type: {activation_type}")
        kwargs = {"dim": 1} if activation_type == "softmax" else {}
        return __supported_activations__[activation_type](**kwargs)

    @staticmethod
    def init_params(params: Dict[str, Any]) -> Dict[str, Any]:
        init = lambda key, value: (  # noqa: E731
            ModelFactory.init_activation(value)  # init activation
            if key == "act_func" or key == "out_func"
            else value
        )
        return {k: init(k, v) for k, v in params.items()}
