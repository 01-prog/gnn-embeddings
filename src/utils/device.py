"""Shared device-resolution helper used by encoders and the training loop.

Centralising the ``"auto"`` → ``torch.device`` logic here prevents the same
one-liner from drifting across multiple modules.
"""

from __future__ import annotations

import torch


def resolve_device(device: str | torch.device) -> torch.device:
    """Resolve a device string to a ``torch.device``.

    Args:
        device: ``"auto"`` (use CUDA if available, else CPU), ``"cpu"``,
            ``"cuda"``, ``"cuda:0"``, etc., or an already-resolved
            ``torch.device``.

    Returns:
        A ``torch.device``.
    """
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
