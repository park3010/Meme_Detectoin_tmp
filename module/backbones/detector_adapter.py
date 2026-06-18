"""Detector adapter for local object and symbol extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.image_utils import image_size
from utils.text_utils import capitalized_spans, keyword_candidates, rhetorical_cues


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
