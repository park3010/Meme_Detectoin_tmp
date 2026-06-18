"""Dataclass schemas for Stage A."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class StageAInput:
    """Input container for internal evidence extraction."""

    sample_id: str
    dataset_name: str
    image_path: str | None
    ocr_text_full: str
    raw_label: int | str | None = None
    annotation: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_sample(cls, sample: dict[str, Any]) -> "StageAInput":
        """Build Stage A input from a dataset sample dictionary."""

        return cls(
            sample_id=str(sample.get("sample_id", "")),
            dataset_name=str(sample.get("dataset_name", "")),
            image_path=sample.get("image_path"),
            ocr_text_full=str(sample.get("ocr_text_full") or sample.get("ocr_text") or ""),
            raw_label=sample.get("raw_label"),
            annotation=sample.get("annotation"),
            metadata=dict(sample.get("metadata") or {}),
        )


@dataclass
class InternalEvidenceItem:
    """One internal evidence token and its provenance."""

    evidence_id: str
    evidence_type: str
    text: str
    score: float
    token_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageAMetadata:
    """Auxiliary weak-cue metadata, not final structured predictions."""

    image_backend: str
    text_backend: str
    roi_count: int
    token_count: int
    tensor_shapes: dict[str, list[int]] = field(default_factory=dict)
    auxiliary_labels: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class StageAOutput:
    """Internal evidence bank and weak cues consumed by later stages."""

    sample_id: str
    dataset_name: str
    internal_tokens: torch.Tensor
    evidence_items: list[InternalEvidenceItem]
    global_visual: torch.Tensor
    global_text: torch.Tensor
    patch_tokens: torch.Tensor
    text_tokens: torch.Tensor
    token_strings: list[str]
    roi_tokens: torch.Tensor
    auxiliary_scores: dict[str, float]
    metadata: StageAMetadata
