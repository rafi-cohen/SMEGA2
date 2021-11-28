import torch
from torch.optim.optimizer import Optimizer, required


class SMEGA2(Optimizer):
    def __init__(self, params, lr=required, momentum_cycle=required, advanced=True, momentum=0, dampening=0, weight_decay=0):
        """
        :param params: Iterable of parameters to optimize or dicts defining parameter groups
        :param lr: Learning rate
        :param momentum_cycle: Number of steps required to update momentum
        :param advanced: Whether to use advanced estimation. If False: Naive estimation
        :param momentum: Momentum factor
        :param dampening: Dampening for momentum
        :param weight_decay: Weight decay
        """
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum_cycle is not required:
            if momentum_cycle <= 0 or not isinstance(momentum_cycle, int):
                raise ValueError("Invalid momentum_cycle value: {}".format(momentum_cycle))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum_cycle=momentum_cycle, advanced=advanced,
                        momentum=momentum, dampening=dampening, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def _get_smega2_state(self, param: torch.Tensor):
        param_state = self.state[param]
        if 'momentum_buffer' not in param_state:
            param_state['momentum_buffer'] = torch.zeros_like(param)
        if 'gradients_buffer' not in param_state:
            param_state['gradients_buffer'] = torch.zeros_like(param)
        if 'step_count' not in param_state:
            param_state['step_count'] = 0
        momentum_buffer = param_state['momentum_buffer']
        gradients_buffer = param_state['gradients_buffer']
        step_count = param_state['step_count']
        return momentum_buffer, gradients_buffer, step_count

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        :param closure: A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum_cycle = group['momentum_cycle']
            momentum = group['momentum']
            dampening = group['dampening']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    momentum_buffer, gradients_buffer, step_count = self._get_smega2_state(p)
                    step_count = param_state['step_count'] = (step_count + 1) % momentum_cycle
                    gradients_buffer.add_(d_p)
                    d_p = d_p.add(momentum_buffer, alpha=momentum)
                    if step_count == 0:
                        momentum_buffer.mul_(momentum).add_(gradients_buffer, alpha=(1 - dampening) / momentum_cycle)
                        gradients_buffer.zero_()

                p.add_(d_p, alpha=-lr)

        return loss

    @torch.no_grad()
    def estimate(self):
        """
        Calculates the estimate parameters `momentum_cycle` steps ahead based on current state
        :return: estimate
        """
        estimate = []
        for group in self.param_groups:
            lr = group['lr']
            momentum_cycle = group['momentum_cycle']
            advanced = group['advanced']
            momentum = group['momentum']
            for p in group['params']:
                momentum_buffer, gradients_buffer, step_count = self._get_smega2_state(p)
                momentum_factor = momentum_cycle
                gradients_factor = 0
                if advanced:
                    momentum_factor += step_count * (momentum - 1)
                    gradients_factor += step_count / momentum_cycle
                momentum_factor *= lr * momentum
                gradients_factor *= lr * momentum
                e = p.sub(gradients_buffer, alpha=gradients_factor).sub_(momentum_buffer, alpha=momentum_factor)
                estimate.append(e)

        return estimate
