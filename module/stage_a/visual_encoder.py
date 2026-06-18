"""Global visual encoder for Stage A."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from module.backbones.clip_wrapper import CLIPWrapper
from utils.tensor_utils import hashed_vector


class GlobalVisualEncoder(nn.Module):
    """Encode image-level and coarse patch-level visual evidence."""

    def __init__(self, hidden_dim: int = 256, prefer_pretrained: bool = False, model_name: str = "ViT-B-32", device: str = "cpu") -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.clip = CLIPWrapper(hidden_dim=hidden_dim, prefer_pretrained=prefer_pretrained, model_name=model_name, device=device)

    def forward(self, image_path: str | Path | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return global image embedding and four coarse patch tokens."""

        global_embedding = self.clip.encode_image(image_path)
        patch_tokens = []
        for idx in range(4):
            patch_hint = hashed_vector(f"patch:{idx}:{image_path}", dim=self.hidden_dim).to(global_embedding.device)
            patch_tokens.append(torch.nn.functional.normalize(0.75 * global_embedding + 0.25 * patch_hint, dim=0))
        return global_embedding, torch.stack(patch_tokens, dim=0)
