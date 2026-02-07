"""Utility functions for CS336 Assignment 1."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random language modeling input-label pairs from dataset.

    For each example in the batch, pick a random starting index and extract
    context_length tokens as input, with the next token shifted by 1 as label.
    """
    max_start = len(dataset) - context_length
    start_indices = np.random.randint(0, max_start, size=(batch_size,))

    x = np.stack([dataset[i : i + context_length] for i in start_indices])
    y = np.stack([dataset[i + 1 : i + 1 + context_length] for i in start_indices])

    x = torch.from_numpy(x).long().to(device)
    y = torch.from_numpy(y).long().to(device)
    return x, y


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Cosine annealing LR schedule with linear warmup.

    - Linear warmup from 0 to max_lr over warmup_iters
    - Cosine decay from max_lr to min_lr over cosine_cycle_iters
    - Constant min_lr after cosine_cycle_iters
    """
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    elif it < cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1.0 + math.cos(math.pi * progress))
    else:
        return min_learning_rate
