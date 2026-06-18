"""Tensor serialization and deterministic feature helpers."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def stable_hash(value: str, modulo: int | None = None) -> int:
    """Return a deterministic integer hash independent of Python hash seed."""

    digest = hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()
    number = int(digest[:16], 16)
    return number % modulo if modulo else number


def hashed_vector(text: str, dim: int = 256, normalize: bool = True) -> torch.Tensor:
    """Create a deterministic dense vector from text using signed hashing."""

    vector = torch.zeros(dim, dtype=torch.float32)
    tokens = text.split() if text else ["<empty>"]
    for token in tokens:
        idx = stable_hash(token, dim)
        sign = 1.0 if stable_hash(f"{token}:sign", 2) == 0 else -1.0
        vector[idx] += sign
    if normalize:
        vector = F.normalize(vector, dim=0)
    return vector


def pad_or_trim_tokens(tokens: torch.Tensor, length: int, dim: int) -> torch.Tensor:
    """Return a fixed-length token matrix."""

    if tokens.numel() == 0:
        tokens = torch.zeros(0, dim)
    if tokens.size(0) >= length:
        return tokens[:length]
    padding = torch.zeros(length - tokens.size(0), dim, dtype=tokens.dtype, device=tokens.device)
    return torch.cat([tokens, padding], dim=0)


def tensor_to_python(value: Any, max_elements: int | None = None) -> Any:
    """Convert dataclasses/tensors/paths into JSON-serializable values."""

    if is_dataclass(value):
        return tensor_to_python(asdict(value), max_elements=max_elements)
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if max_elements is not None and tensor.numel() > max_elements:
            return {
                "shape": list(tensor.shape),
                "preview": tensor.flatten()[:max_elements].tolist(),
            }
        return tensor.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): tensor_to_python(v, max_elements=max_elements) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_to_python(v, max_elements=max_elements) for v in value]
    return value
