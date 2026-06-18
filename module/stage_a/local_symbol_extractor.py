"""Local object and symbol extraction for Stage A."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from module.backbones.clip_wrapper import CLIPWrapper
from module.backbones.detector_adapter import Detection, DetectorAdapter


class LocalObjectSymbolExtractor(nn.Module):
    """Extract region-level evidence with a detector adapter and ROI encoder."""

    def __init__(
        self,
        hidden_dim: int = 256,
        detector_mode: str = "heuristic",
        max_rois: int = 3,
        prefer_pretrained_clip: bool = False,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.detector = DetectorAdapter(mode=detector_mode, max_rois=max_rois)
        self.roi_encoder = CLIPWrapper(hidden_dim=hidden_dim, prefer_pretrained=prefer_pretrained_clip, device=device)

    def forward(self, image_path: str | Path | None, ocr_text: str) -> tuple[list[Detection], torch.Tensor, list[dict[str, Any]]]:
        """Return detections, ROI tokens, and serializable metadata."""

        detections = self.detector.detect(image_path, ocr_text)
        boxes = [detection.box for detection in detections]
        tokens = self.roi_encoder.encode_rois(image_path, boxes)
        metadata = [
            {
                "box": list(detection.box),
                "label": detection.label,
                "score": detection.score,
                "metadata": detection.metadata,
            }
            for detection in detections
        ]
        return detections, tokens, metadata
