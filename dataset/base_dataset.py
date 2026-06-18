"""Base dataset primitives for meme data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MemeSample:
    """A dataset-agnostic meme sample returned by the unified loader."""

    sample_id: str
    dataset_name: str
    image_path: str | None
    ocr_text_full: str = ""
    raw_label: int | str | None = None
    annotation: dict[str, Any] | None = None
    raw_record: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary for stage inputs and serialization."""

        return {
            "sample_id": self.sample_id,
            "dataset_name": self.dataset_name,
            "image_path": self.image_path,
            "ocr_text_full": self.ocr_text_full,
            "raw_label": self.raw_label,
            "annotation": self.annotation,
            "raw_record": self.raw_record,
            "metadata": self.metadata,
        }


class BaseMemeDataset:
    """Small base class with preview and statistics hooks."""

    samples: list[MemeSample]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index].to_dict()

    def preview(self, n: int = 3) -> list[dict[str, Any]]:
        """Return compact sample previews."""

        previews: list[dict[str, Any]] = []
        for sample in self.samples[:n]:
            previews.append(
                {
                    "sample_id": sample.sample_id,
                    "dataset_name": sample.dataset_name,
                    "image_path": sample.image_path,
                    "ocr_text_full": sample.ocr_text_full[:160],
                    "raw_label": sample.raw_label,
                    "has_annotation": sample.annotation is not None,
                }
            )
        return previews

    def validate_files(self) -> dict[str, int]:
        """Count missing images and empty OCR strings."""

        missing_images = 0
        empty_text = 0
        for sample in self.samples:
            if sample.image_path is None or not Path(sample.image_path).exists():
                missing_images += 1
            if not sample.ocr_text_full.strip():
                empty_text += 1
        return {"missing_images": missing_images, "empty_text": empty_text}
