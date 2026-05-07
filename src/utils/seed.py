"""Reproducibility helpers: a single ``set_seed`` function used by all entry
points (training scripts, tests, the evaluation harness) to make a run
deterministic across ``random``, ``numpy``, and ``torch`` (CPU + CUDA)."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed ``random``, ``numpy``, and ``torch`` (CPU and CUDA) for reproducibility.

    Args:
        seed: Non-negative integer seed.
        deterministic: If True, also force cuDNN into deterministic mode.
            This may slow down training slightly but is required for the
            "5-seed mean ± std" evaluation protocol used in the project.

    Returns:
        None.

    Raises:
        ValueError: If ``seed`` is negative.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed(42)
    a = torch.rand(3)
    set_seed(42)
    b = torch.rand(3)
    assert torch.allclose(a, b), "set_seed is not deterministic"
    print("set_seed OK:", a.tolist())
