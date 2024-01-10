import torch

class MeZO(torch.optim.Optimizer):
    """
    Minimal implementation of
        Fine-Tuning Language Models with Just Forward Passes
        https://arxiv.org/abs/2305.17333
    
    Args:
        opt: torch.optim.Optimizer
        eps: perturbation size for mezo    
    """

    def __init__(self, opt, eps=0.01):
        self.opt = opt
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
        self.perturb_parameters(eps_scale=1)
        
        try:
            with torch.no_grad():
                loss_pos = closure()
        except RuntimeError as e:
            raise RuntimeError(str(e) + '. Hint: ensure that backward is disbaled inside closure.')
        
        self.perturb_parameters(eps_scale=-2)
        
        with torch.no_grad():
            loss_neg = closure()

        self.perturb_parameters(eps_scale=1)
        proj_grad = (loss_pos - loss_neg) / (2 * self.eps)
        
        self.backward(proj_grad)
        self.opt.step()
        return (loss_pos + loss_neg) / 2.
    
    def zero_grad(self):
        """ clears optimizer grad """
        return self.opt.zero_grad()
    
    def backward(self, proj_grad, only_requires_grad=True):
        """ mezo backward using projected gradient """
        torch.manual_seed(self.seed)
        for p in self.params:
            if only_requires_grad and not p.requires_grad:
                continue                
            p.grad = proj_grad * torch.randn_like(p) * self.eps_sigma
        return
    
    @property
    def params(self):
        """ get all parameters by flattening param_groups """
        return [param for param_group in self.param_groups for param in param_group['params']]

    def perturb_parameters(self, eps_scale, only_requires_grad=True):
        """ perturb parameters """
        torch.manual_seed(self.seed)
        for p in self.params:
            if only_requires_grad and not p.requires_grad:
                continue
            p.data += eps_scale * self.eps * torch.randn_like(p.data) * self.eps_sigma
        return
    