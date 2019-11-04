r'''
Implementation of rectified Adam (RAdam)
written for Pytorch. The original paper
can be found at https://arxiv.org/pdf/1908.03265.pdf.
'''
# standard library imports
import math

# third party imports
import torch
from torch.optim.optimizer import Optimizer, required


class RAdam(Optimizer):
    '''
    Class to implement a torch optimizer for RAdam.
    '''

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0, ams_grad=False):
        '''
        Initializes an instance of the RAdam class.

        Arguments
        ---------
            params : (iterable)
                Iterable of parameters to optimize or dicts defining parameter groups.
            lr : (float, optional, default=1e-3)
                Learning rate
            betas : (Tuple[float, float], optional, default=(0.9, 0.999))
                Coefficients used for computing running averages of gradient and its square.
            eps : (float, optional, default=1e-8)
                Term added to the denominator to improve numerical stability.
            weight_decay : (float, optional, default=0)
                Weight decay (L2 penalty)
            ams_grad : (bool, optional, default=False)
        '''
        if not 0.0 <= lr:
            raise ValueError(f'Invalid learning rate: {lr}')
        if not 0.0 <= eps:
            raise ValueError(f'Invalid epsilon value: {eps}')
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f'Invalid beta parameters at index 0: {betas[0]}')
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f'Invalid beta parameters at index 1: {betas[1]}')
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, ams_grad=ams_grad)
        super(RAdam, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
    
    def step(self, closure=None):
        '''
        Performs a single optimization step.

        Arguments
        ---------
            closure : (callable, optional)
                A closure that reevaluates the model and returns the loss.
        '''
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintaining max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                
                step_size = group['lr'] / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)
            
        return loss