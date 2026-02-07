"""Optimizer and gradient utilities for CS336 Assignment 1."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Clip combined gradients to have L2 norm at most max_l2_norm.

    Modifies parameter.grad in-place.
    """
    parameters = list(parameters)
    # Compute total L2 norm of all gradients
    total_norm_sq = 0.0
    for p in parameters:
        if p.grad is not None:
            total_norm_sq += p.grad.data.norm(2).item() ** 2
    total_norm = math.sqrt(total_norm_sq)

    if total_norm > max_l2_norm:
        scale = max_l2_norm / total_norm
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(scale)


class AdamW(torch.optim.Optimizer):
    """AdamW optimizer with decoupled weight decay."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data

                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                state["step"] += 1
                t = state["step"]

                m = state["m"]
                v = state["v"]

                # Update biased first and second moment estimates
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                # Parameter update (Adam step)
                p.data.add_(m_hat / (v_hat.sqrt() + eps), alpha=-lr)

                # Decoupled weight decay
                p.data.add_(p.data, alpha=-lr * weight_decay)

        return loss
