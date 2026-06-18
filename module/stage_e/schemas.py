"""Dataclass schemas for Stage E."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class Prediction:
    """A labeled prediction with score distribution."""

    label: str
    score: float
    scores: dict[str, float] = field(default_factory=dict)
    logits: torch.Tensor | None = None
    labels: list[str] = field(default_factory=list)
    auxiliary: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageEMetadata:
    """Stage E metadata."""

    internal_evidence_count: int
    external_evidence_count: int
    rationale_backend: str = "template"


@dataclass
class StageEOutput:
    """Final structured interpretation output."""

    sample_id: str
    dataset_name: str
    harmfulness: Prediction
    target: Prediction
    intent: Prediction
    tactic: Prediction
    supporting_evidence: dict[str, list[dict[str, Any]]]
    rationale: str
    structured_prediction: dict[str, Any]
    metadata: StageEMetadata
