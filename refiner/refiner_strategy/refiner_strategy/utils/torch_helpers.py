"""Device selection helper for PyTorch.

Factored out so every module picks the same device without duplicating
the availability checks.
"""
from __future__ import annotations


def select_device() -> str:
    """Return the best available torch device: cuda > mps > cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
