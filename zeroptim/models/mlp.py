from typing import Optional
import torch.nn as nn
import torch


class MLP(nn.Module):

    def __init__(
        self,
        input_dim: int, 
        hidden_dims: list[int],
        output_dim: int,
        act_func: Optional[nn.Module] = None,
        out_func: Optional[nn.Module] = None,
        p_dropout: Optional[float] = 0.0,
        batch_norm: Optional[bool] = False,
        bias: Optional[bool] = True,
        init: Optional[str] = 'xavier_uniform',
        seed: Optional[int] = None
    ) -> None:
        super(MLP, self).__init__()
        assert len(hidden_dims) >= 1

        if seed is not None:
            torch.manual_seed(seed)  # Set the seed

        self.input_dim = input_dim
        self.output_dim = output_dim
        # Trick to create all hidden layers at once
        self.hidden_dims = [input_dim] + hidden_dims

        # Activation function on hidden layers (non-linearity)
        self.act_func = act_func if act_func else nn.Identity()
        # Output function on logits layer (post-processing)
        self.out_func = out_func if out_func else nn.Identity()
        
        self.bias = bias # Whether to use biases (address underfitting issues)
        self.p_dropout = p_dropout # Dropout probability (address overfitting issues)
        self.batch_norm = batch_norm # Whether to use BN (address internal covariate shift)
        self.init_method = init # Initialization method for weights and biases

        self.layers = nn.Sequential(
            # nn.Flatten(),
            *[nn.Sequential(
                nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1], bias=self.bias),
                *[nn.BatchNorm1d(self.hidden_dims[i+1])] if self.batch_norm else [],
                self.act_func,
                *[nn.Dropout(self.p_dropout)] if self.p_dropout > 0 else [],
            ) for i in range(len(self.hidden_dims) - 1)],
            # Logits layer separate: No additional processing modules
            nn.Linear(self.hidden_dims[-1], self.output_dim, bias=self.bias),
        )

        # Initialize weights and biases
        for m in self.layers.modules():
            if isinstance(m, nn.Linear):
                if self.init_method == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight)
                elif self.init_method == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight)
                elif self.init_method == 'he_normal':
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                elif self.init_method == 'he_uniform':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif self.init_method == 'zeros':
                    nn.init.zeros_(m.weight)
                elif self.init_method == 'ones':
                    nn.init.ones_(m.weight)
            
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x):
        logits = self.layers(x)
        return self.out_func(logits)
    
    
    def norm(self):
        return sum(p.norm() for p in self.layers.parameters() if p.requires_grad)


    def count_params(self):
        return sum(p.numel() for p in self.layers.parameters() if p.requires_grad)


    def params(self):
        params_list = [] # (weights, biases, BNs)
        
        # Iterate over child modules in MLP block
        for module in self.layers.modules():
            if isinstance(module, nn.Linear):
                for param in module.parameters():
                    params_list.append(param)
            elif isinstance(module, nn.BatchNorm1d) and self.batch_norm:
                for param in module.parameters():
                    params_list.append(param)

        return params_list
    
    
    def weights(self):
        weights_list = []
        
        # Iterate over child modules in MLP block
        for module in self.layers.modules():
            if isinstance(module, nn.Linear):
                weights_list.append(module.weight)

        return weights_list
    

    def biases(self):
        biases_list = []
        
        # Iterate over child modules in MLP block
        for module in self.layers.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                biases_list.append(module.bias)

        return biases_list


    def bns(self):
        bns_list = []
        
        # Iterate over child modules in MLP block
        for module in self.layers.modules():
            if isinstance(module, nn.BatchNorm1d):
                bns_list.append(module)

        return bns_list
    

    def clone(self):
        # Create a new instance of MLP with the same configuration
        m_ = MLP(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims[1:],  # exclude the first element as it's input_dim
            output_dim=self.output_dim,
            act_func=self.act_func,
            out_func=self.out_func,
            p_dropout=self.p_dropout,
            batch_norm=self.batch_norm,
            bias=self.bias,
            init=self.init_method,
            seed=None  # we don't want to replicate the seed (as already initialized)
        )

        # Copy the parameters and buffers from the current model
        m_.load_state_dict(self.state_dict())

        return m_
    
