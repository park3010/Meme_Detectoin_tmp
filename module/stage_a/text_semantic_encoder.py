"""Text semantic encoder for Stage A."""

from __future__ import annotations

import torch
from torch import nn

from module.backbones.text_encoder_wrapper import TextEncoderWrapper


class TextSemanticEncoder(nn.Module):
    """Encode OCR text into global and token-level semantic states."""

    def __init__(
        self,
        hidden_dim: int = 256,
        prefer_transformers: bool = False,
        model_name: str = "microsoft/deberta-v3-base",
        max_tokens: int = 64,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.encoder = TextEncoderWrapper(
            hidden_dim=hidden_dim,
            prefer_transformers=prefer_transformers,
            model_name=model_name,
            max_tokens=max_tokens,
            device=device,
        )

    def forward(self, text: str) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        """Return global text embedding, token embeddings, and token strings."""

        return self.encoder.encode(text)
