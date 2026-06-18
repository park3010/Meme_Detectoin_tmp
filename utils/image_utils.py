"""Image loading and fallback feature extraction."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from utils.tensor_utils import hashed_vector


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def load_image(path: str | Path | None):
    """Load an RGB PIL image when PIL and the file are available."""

    if path is None:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    try:
        from PIL import Image

        return Image.open(file_path).convert("RGB")
    except Exception:
        return None


def fallback_image_embedding(path: str | Path | None, dim: int = 256) -> torch.Tensor:
    """Build a deterministic image embedding from pixels, or path text if unavailable."""

    image = load_image(path)
    if image is None:
        return hashed_vector(f"missing-image:{path}", dim=dim)
    try:
        small = image.resize((16, 16))
        values = torch.tensor(list(small.getdata()), dtype=torch.float32).flatten() / 255.0
        if values.numel() < dim:
            values = F.pad(values, (0, dim - values.numel()))
        chunks = values[: dim * max(1, values.numel() // dim)].reshape(dim, -1)
        vector = chunks.mean(dim=1)
        return F.normalize(vector, dim=0)
    except Exception:
        return hashed_vector(f"image:{path}", dim=dim)


def image_size(path: str | Path | None) -> tuple[int, int] | None:
    """Return image size as (width, height) when available."""

    image = load_image(path)
    if image is None:
        return None
    return image.size
