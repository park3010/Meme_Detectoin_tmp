"""Collation utilities for PyTorch DataLoader usage."""

from __future__ import annotations

from typing import Any


def meme_collate_fn(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate variable-size meme samples into lists keyed by field name."""

    if not samples:
        return {}
    keys = sorted({key for sample in samples for key in sample})
    return {key: [sample.get(key) for sample in samples] for key in keys}


normalized_meme_collate_fn = meme_collate_fn
