from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                state = self.state[p]

                # TODO: implement state creation and parameters initialization: step, momentum_t, rms_t, beta_1, beta_2
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_t"] = torch.zeros_like(p.data)
                    state["rms_t"] = torch.zeros_like(p.data)

                beta_1, beta_2 = group["betas"]
                momentum_t = state["momentum_t"]
                rms_t = state["rms_t"]
                state["step"] += 1


                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]

                # TODO: update first and second moments of the gradients
                momentum_t.mul_(beta_1).add_(grad, alpha=1 - beta_1)
                rms_t.mul_(beta_2).addcmul_(grad, grad, value=1 - beta_2)

                # TODO: Bias correction
# Please note that we are using the "efficient version" given in
# https://arxiv.org/abs/1412.6980
                if group["correct_bias"]:
                    alpha_t = alpha * ((1 - beta_2 ** state["step"]) ** 0.5) / (1 - beta_1 ** state["step"])
                else:
                    alpha_t = alpha

                # Update parameters
                # params = p - alpha_t * momentum_t / (torch.sqrt(rms_t) + group['eps'])
                p.data.addcdiv_(momentum_t, torch.sqrt(rms_t) + group['eps'], value=-alpha_t)

                # TODO: Add weight decay after the main gradient-based updates
# Please note that the learning rate should be incorporated into this update
                if group["weight_decay"] > 0:
                    p.data.add_(p.data, alpha=-alpha * group["weight_decay"])

        return loss
