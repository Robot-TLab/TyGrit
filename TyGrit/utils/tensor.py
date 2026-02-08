"""Torch ↔ NumPy conversion helpers."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch


def to_numpy(x: torch.Tensor | npt.NDArray) -> np.ndarray:
    """Convert a torch tensor (possibly batched) to a 1-D numpy array.

    ManiSkill3 returns batched GPU tensors with shape ``(1, N)``.
    This helper detaches, moves to CPU, and squeezes the batch dimension.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if x.ndim == 2:
        x = x[0]
    return np.asarray(x)
