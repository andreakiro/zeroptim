from typing import Optional
from collections import OrderedDict
import torch.nn as nn
import torch


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        act_func: Optional[nn.Module] = None,
        bias: bool = True,
        use_batch_norm: bool = False,
        p_dropout: float = 0.0,
    ):
        super(LinearBlock, self).__init__()

        modules = OrderedDict()
        modules["linear"] = nn.Linear(
            in_features=in_features, out_features=out_features, bias=bias
        )
        # Batch norm requires B > 1 (otherwise ill defined) + not recommended with dropout
        modules["bn"] = (
            nn.BatchNorm1d(num_features=out_features)
            if use_batch_norm
            else nn.Identity()
        )
        modules["act_func"] = act_func if act_func else nn.Identity()
        modules["dropout"] = nn.Dropout(p=p_dropout)

        self.block = nn.Sequential(modules)

    def forward(self, x):
        return self.block(x)


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: list[int],
        out_features: int,
        act_func: Optional[nn.Module] = None,
        out_func: Optional[nn.Module] = None,
        bias: bool = True,
        use_batch_norm: bool = False,
        p_dropout: float = 0.0,
        init: str = "xavier_uniform",
        seed: Optional[int] = None,
    ) -> None:
        super(MLP, self).__init__()
        assert len(hidden_features) >= 1

        if seed is not None:
            # seed for reproducibility
            torch.manual_seed(seed)

        self.out_func = out_func if out_func else nn.Identity()

        layers = OrderedDict()
        layers["flatten"] = nn.Flatten()

        hidden_dims = [in_features] + hidden_features

        for i in range(len(hidden_dims) - 1):
            layers[f"block-{i}"] = LinearBlock(
                in_features=hidden_dims[i],
                out_features=hidden_dims[i + 1],
                act_func=act_func,
                bias=bias,
                use_batch_norm=use_batch_norm,
                p_dropout=p_dropout,
            )

        layers["fc"] = nn.Linear(
            in_features=hidden_dims[-1],
            out_features=out_features,
            bias=bias,
        )

        self.layers = nn.Sequential(layers)
        self.init_weights(init)
        self.penultimate = None

    def init_weights(self, init_method) -> None:
        for layer in self.layers:
            if isinstance(layer, LinearBlock):
                for submodule in layer.modules():
                    if isinstance(submodule, nn.Linear):
                        self.apply_init(submodule, init_method)
            elif isinstance(layer, nn.Linear):
                self.apply_init(layer, init_method)

    def apply_init(self, linear_layer, init_method):
        if init_method == "xavier_uniform":
            nn.init.xavier_uniform_(linear_layer.weight)
        elif init_method == "xavier_normal":
            nn.init.xavier_normal_(linear_layer.weight)
        elif init_method == "he_normal":
            nn.init.kaiming_normal_(linear_layer.weight, nonlinearity="relu")
        elif init_method == "he_uniform":
            nn.init.kaiming_uniform_(linear_layer.weight, nonlinearity="relu")

    def forward(self, x):
        self.penultimate = self.layers[:-1](x)
        clf = self.layers[-1](self.penultimate)
        logits = self.out_func(clf)
        return logits
