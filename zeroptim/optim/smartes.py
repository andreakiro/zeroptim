import torch

class SmartES(torch.optim.Optimizer):
    """ Smart Evolution Strategies (SmartES) optimizer.
    
    Args:
        opt: torch.optim.Optimizer
        eps: perturbation size  
    """

    def __init__(self, opt, model, eps=0.01):
        self.opt = opt
        self.model = model
        self.eps = eps
        self.eps_sigma = 1 # experimental feature to scale variance of noise
        self.max_seed = 10**6
        self.step_count = 0
        
        super().__init__(self.opt.param_groups, {'eps': self.eps, 'eps_var': self.eps_sigma})
        
        self.param_groups = self.opt.param_groups
        self.defaults.update(self.opt.defaults)
        return
    
    def step(self, closure=None):
        """ full mezo training step """
        if closure is None:
            raise RuntimeError('MeZO optimizer expected closure but not provided.')

        self.seed = torch.randint(self.max_seed, (1,)).item() + self.step_count
        self.step_count += 1
        
        rm_hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                rm_hooks.append(module.register_forward_hook(self.build_hook(1)))
        try:
            with torch.no_grad():
                loss_pos = closure()
        except RuntimeError as e:
            raise RuntimeError(str(e) + '. Hint: ensure that backward is disbaled inside closure.')
    
        for rm_hook in rm_hooks:
            rm_hook.remove()

        
        rm_hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                rm_hooks.append(module.register_forward_hook(self.build_hook(-1)))
        
        with torch.no_grad():
            loss_neg = closure()
        
        for rm_hook in rm_hooks:
            rm_hook.remove()

        proj_grad = (loss_pos - loss_neg) / (2 * self.eps)

        self.backward(proj_grad, closure)

        
        self.opt.step()
        return (loss_pos + loss_neg) / 2.
    
    def zero_grad(self):
        """ clears optimizer grad """
        return self.opt.zero_grad()
    
    def backward(self, proj_grad, closure, only_requires_grad=True):
        """ mezo backward using projected gradient """
        torch.manual_seed(self.seed)

        rm_hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                rm_hooks.append(module.register_forward_hook(self.build_hook(proj_grad, update_param=True)))

        loss = closure()

        for rm_hook in rm_hooks:
            rm_hook.remove()

        return loss
    
    @property
    def params(self):
        """ get all parameters by flattening param_groups """
        return [param for param_group in self.param_groups for param in param_group['params']]
    
    def build_hook(self, eps_scale, update_param=False):
        def forward_hook(module, input, output):
            assert len(input) == 1

            torch.manual_seed(self.seed)
            act_perturb = torch.randn_like(output.data)

            if update_param:
                if len(input[0].shape) == 3:
                    x = input[0]
                    z = torch.outer(act_perturb, x )
                elif len(input[0].shape) == 2:
                    x = input[0]
                    z = x.T@act_perturb/x.shape[0]
                else:
                    raise NotImplementedError

                module.weight.grad = eps_scale * z.T * self.eps_sigma
            else:
                if len(input[0].shape) == 3:
                    x = input[0]
                elif len(input[0].shape) == 2:
                    x = input[0]
                else:
                    raise NotImplementedError

                output.data += eps_scale * self.eps * (x@x.T@act_perturb)/x.shape[0] * self.eps_sigma
        
        return forward_hook