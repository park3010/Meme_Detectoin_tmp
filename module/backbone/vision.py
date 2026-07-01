"""Vision backbone adapters for Stage A and baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from utils.image_utils import fallback_image_embedding, image_size
from utils.logging_utils import setup_logger
from utils.tensor_utils import hashed_vector
from utils.text_utils import capitalized_spans, keyword_candidates, rhetorical_cues


# =============================================================================
# CLIP image wrapper
# =============================================================================

logger = setup_logger(__name__)


class CLIPWrapper(nn.Module):
    """Expose a stable image encoding API while optional CLIP deps are absent."""

    def __init__(
        self,
        hidden_dim: int = 256,
        prefer_pretrained: bool = False,
        model_name: str = "ViT-B-32",
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.device_name = device
        self.model: Any | None = None
        self.preprocess: Any | None = None
        self.backend = "fallback"
        self._projection: nn.Linear | None = None
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)
        if prefer_pretrained:
            self._try_load_clip(model_name, device)

    def _try_load_clip(self, model_name: str, device: str) -> None:
        try:
            import open_clip  # type: ignore

            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=None)
            self.model = model.to(device).eval()
            self.preprocess = preprocess
            self.backend = "open_clip"
            logger.info("Using open_clip backend: %s", model_name)
            return
        except Exception as exc:
            logger.info("open_clip unavailable, trying clip package: %s", exc)
        try:
            import clip  # type: ignore

            model, preprocess = clip.load(model_name, device=device)
            self.model = model.eval()
            self.preprocess = preprocess
            self.backend = "clip"
            logger.info("Using clip backend: %s", model_name)
        except Exception as exc:
            logger.info("CLIP unavailable; using fallback image encoder: %s", exc)

    @torch.no_grad()
    def encode_image(self, image_path: str | Path | None) -> torch.Tensor:
        """Encode a whole image to the configured hidden dimension."""

        target_device = self._target_device()
        if self.model is not None and self.preprocess is not None and image_path is not None:
            try:
                from PIL import Image

                image = Image.open(image_path).convert("RGB")
                tensor = self.preprocess(image).unsqueeze(0).to(self.device_name)
                feature = self.model.encode_image(tensor).float().squeeze(0).to(target_device)
                return self._project_feature(feature)
            except Exception as exc:
                logger.warning("CLIP image encoding failed for %s; falling back: %s", image_path, exc)
        return fallback_image_embedding(image_path, dim=self.hidden_dim).to(target_device)

    @torch.no_grad()
    def encode_rois(self, image_path: str | Path | None, boxes: list[tuple[float, float, float, float]]) -> torch.Tensor:
        """Encode ROI boxes; fallback appends box coordinates to the image identity."""

        target_device = self._target_device()
        if not boxes:
            return torch.zeros(0, self.hidden_dim, device=target_device)
        vectors = []
        for box in boxes:
            box_text = " ".join(f"{value:.3f}" for value in box)
            base = self.encode_image(image_path)
            roi_hint = hashed_vector(f"roi:{image_path}:{box_text}", dim=self.hidden_dim).to(base.device)
            vectors.append(F.normalize(0.7 * base + 0.3 * roi_hint, dim=0))
        return torch.stack(vectors, dim=0)

    def _project_feature(self, feature: torch.Tensor) -> torch.Tensor:
        feature = feature.flatten().float()
        if feature.numel() == self.hidden_dim:
            return F.normalize(feature, dim=0)
        if self._projection is None or self._projection.in_features != feature.numel():
            self._projection = nn.Linear(feature.numel(), self.hidden_dim).to(feature.device)
        else:
            self._projection = self._projection.to(feature.device)
        projected = self._projection(feature.unsqueeze(0)).squeeze(0)
        return F.normalize(projected, dim=0)

    def _target_device(self) -> torch.device:
        for param in self.parameters():
            return param.device
        for buffer in self.buffers():
            return buffer.device
        try:
            return torch.device(self.device_name)
        except (TypeError, RuntimeError):
            return torch.device("cpu")


# =============================================================================
# Detector adapter
# =============================================================================

@dataclass
class Detection:
    """A local region proposal or symbolic cue."""

    box: tuple[float, float, float, float]
    label: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class DetectorAdapter:
    """Phase-1 detector adapter with an explicit fallback mode."""

    def __init__(self, mode: str = "heuristic", max_rois: int = 3) -> None:
        self.mode = mode
        self.max_rois = max_rois

    def detect(self, image_path: str | Path | None, ocr_text: str = "") -> list[Detection]:
        """Return detections; heuristic mode creates broad pseudo-regions."""

        if self.mode == "empty":
            return []
        size = image_size(image_path)
        entities = capitalized_spans(ocr_text, limit=self.max_rois)
        cues = list(rhetorical_cues(ocr_text).keys())
        labels = [*entities, *cues, *keyword_candidates(ocr_text, limit=self.max_rois)] or ["whole_image"]
        detections: list[Detection] = []
        if size is None:
            for idx, label in enumerate(labels[: self.max_rois]):
                detections.append(
                    Detection(
                        box=(0.0, 0.0, 1.0, 1.0),
                        label=f"pseudo_{label}",
                        score=max(0.25, 0.55 - idx * 0.08),
                        metadata={"fallback": True},
                    )
                )
            return detections

        width, height = size
        templates = [
            (0.0, 0.0, width, height),
            (0.0, 0.0, width, height * 0.45),
            (0.0, height * 0.55, width, height),
            (0.0, height * 0.25, width * 0.5, height * 0.75),
            (width * 0.5, height * 0.25, width, height * 0.75),
        ]
        for idx, label in enumerate(labels[: self.max_rois]):
            box = templates[idx % len(templates)]
            detections.append(
                Detection(
                    box=tuple(float(v) for v in box),
                    label=f"region_{label}",
                    score=max(0.3, 0.7 - idx * 0.1),
                    metadata={"image_size": [width, height], "fallback": True},
                )
            )
        return detections


__all__ = ["CLIPWrapper", "Detection", "DetectorAdapter"]
