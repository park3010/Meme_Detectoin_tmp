"""Simple harmfulness baseline models."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from module.backbone.text import TextEncoderWrapper
from module.backbone.vision import CLIPWrapper
class MLPClassifierHead(nn.Module):
    """Two-layer MLP classifier used by all harmfulness baselines."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_classes: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features.float())


def classifier_output(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return logits, probabilities, and harmfulness probability."""

    probs = torch.softmax(logits, dim=-1)
    return {
        "logits": logits,
        "probabilities": probs,
        "prob_harmful": probs[:, 1],
    }


class ImageOnlyCLIPClassifier(nn.Module):
    """Classify harmfulness from meme image features only."""

    model_name = "image_only_clip"

    def __init__(
        self,
        hidden_dim: int = 256,
        prefer_pretrained_clip: bool = False,
        clip_model_name: str = "ViT-B-32",
        device: str = "cpu",
        clip_pretrained_tag: str | None = None,
        clip_checkpoint_path: str | None = None,
        clip_cache_dir: str | None = None,
        clip_local_files_only: bool = True,
        clip_allow_download: bool = False,
        clip_asset_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = CLIPWrapper(
            hidden_dim=hidden_dim,
            prefer_pretrained=prefer_pretrained_clip,
            model_name=clip_model_name,
            device=device,
            pretrained_tag=clip_pretrained_tag,
            checkpoint_path=clip_checkpoint_path,
            cache_dir=clip_cache_dir,
            local_files_only=clip_local_files_only,
            allow_download=clip_allow_download,
            asset_mode=clip_asset_mode,
        )
        self.classifier = MLPClassifierHead(hidden_dim, hidden_dim=hidden_dim)

    def forward(self, image_paths: list[str | None], ocr_texts: list[str] | None = None) -> dict[str, torch.Tensor]:
        features = torch.stack([self.encoder.encode_image(path) for path in image_paths], dim=0)
        features = features.to(next(self.classifier.parameters()).device)
        return classifier_output(self.classifier(features))


class TextOnlyEncoderClassifier(nn.Module):
    """Classify harmfulness from OCR text features only."""

    model_name = "text_only_encoder"

    def __init__(
        self,
        hidden_dim: int = 256,
        prefer_transformers: bool = False,
        text_model_name: str = "microsoft/deberta-v3-base",
        device: str = "cpu",
        text_checkpoint_path: str | None = None,
        text_cache_dir: str | None = None,
        text_local_files_only: bool = True,
        text_allow_download: bool = False,
        text_asset_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = TextEncoderWrapper(
            hidden_dim=hidden_dim,
            prefer_transformers=prefer_transformers,
            model_name=text_model_name,
            device=device,
            checkpoint_path=text_checkpoint_path,
            cache_dir=text_cache_dir,
            local_files_only=text_local_files_only,
            allow_download=text_allow_download,
            asset_mode=text_asset_mode,
        )
        self.classifier = MLPClassifierHead(hidden_dim, hidden_dim=hidden_dim)

    def forward(self, image_paths: list[str | None] | None = None, ocr_texts: list[str] | None = None) -> dict[str, torch.Tensor]:
        texts = ocr_texts or ["" for _ in image_paths or []]
        features = torch.stack([self.encoder.encode(text)[0] for text in texts], dim=0)
        features = features.to(next(self.classifier.parameters()).device)
        return classifier_output(self.classifier(features))


class CLIPTextConcatClassifier(nn.Module):
    """Classify harmfulness from concatenated image and OCR text embeddings."""

    model_name = "clip_text_concat"

    def __init__(
        self,
        hidden_dim: int = 256,
        prefer_pretrained_clip: bool = False,
        prefer_transformers: bool = False,
        clip_model_name: str = "ViT-B-32",
        text_model_name: str = "microsoft/deberta-v3-base",
        device: str = "cpu",
        clip_pretrained_tag: str | None = None,
        clip_checkpoint_path: str | None = None,
        clip_cache_dir: str | None = None,
        clip_local_files_only: bool = True,
        clip_allow_download: bool = False,
        text_checkpoint_path: str | None = None,
        text_cache_dir: str | None = None,
        text_local_files_only: bool = True,
        text_allow_download: bool = False,
        clip_asset_mode: str | None = None,
        text_asset_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.image_encoder = CLIPWrapper(
            hidden_dim=hidden_dim,
            prefer_pretrained=prefer_pretrained_clip,
            model_name=clip_model_name,
            device=device,
            pretrained_tag=clip_pretrained_tag,
            checkpoint_path=clip_checkpoint_path,
            cache_dir=clip_cache_dir,
            local_files_only=clip_local_files_only,
            allow_download=clip_allow_download,
            asset_mode=clip_asset_mode,
        )
        self.text_encoder = TextEncoderWrapper(
            hidden_dim=hidden_dim,
            prefer_transformers=prefer_transformers,
            model_name=text_model_name,
            device=device,
            checkpoint_path=text_checkpoint_path,
            cache_dir=text_cache_dir,
            local_files_only=text_local_files_only,
            allow_download=text_allow_download,
            asset_mode=text_asset_mode,
        )
        self.classifier = MLPClassifierHead(hidden_dim * 2, hidden_dim=hidden_dim)

    def forward(self, image_paths: list[str | None], ocr_texts: list[str] | None = None) -> dict[str, torch.Tensor]:
        texts = ocr_texts or ["" for _ in image_paths]
        image_features = torch.stack([self.image_encoder.encode_image(path) for path in image_paths], dim=0)
        text_features = torch.stack([self.text_encoder.encode(text)[0] for text in texts], dim=0)
        features = torch.cat([image_features, text_features], dim=-1)
        features = features.to(next(self.classifier.parameters()).device)
        return classifier_output(self.classifier(features))


__all__ = [
    "MLPClassifierHead",
    "classifier_output",
    "ImageOnlyCLIPClassifier",
    "TextOnlyEncoderClassifier",
    "CLIPTextConcatClassifier",
]


BASELINE_REGISTRY = {
    "image_only_clip": ImageOnlyCLIPClassifier,
    "text_only_encoder": TextOnlyEncoderClassifier,
    "clip_text_concat": CLIPTextConcatClassifier,
}


def build_baseline(name: str, config: dict[str, Any] | None = None) -> nn.Module:
    """Build a baseline classifier by registry name."""

    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unsupported baseline model: {name}")
    return BASELINE_REGISTRY[name](**dict(config or {}))


__all__ = [
    "MLPClassifierHead",
    "classifier_output",
    "ImageOnlyCLIPClassifier",
    "TextOnlyEncoderClassifier",
    "CLIPTextConcatClassifier",
    "BASELINE_REGISTRY",
    "build_baseline",
]
