"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import torch


def set_seed(seed: int = 13) -> None:
    """Seed Python and PyTorch without assuming CUDA is available."""

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    # Avoid probing CUDA on CPU-only/default runs; some research machines have
    # mismatched drivers that warn during availability checks.
    if os.environ.get("MEME_DETECTION_SEED_CUDA", "").lower() in {"1", "true", "yes"}:
        torch.cuda.manual_seed_all(seed)
