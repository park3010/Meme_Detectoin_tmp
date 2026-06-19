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
    """Auditable structured-output provenance metadata."""

    internal_evidence_count: int
    external_evidence_count: int
    rationale_backend: str = "template"
    prediction_fields: list[str] = field(default_factory=list)
    field_provenance: dict[str, str] = field(default_factory=dict)
    label_spaces: dict[str, list[str]] = field(default_factory=dict)
    trainable_logits_fields: list[str] = field(default_factory=list)
    proxy_fields: list[str] = field(default_factory=list)
    template_fields: list[str] = field(default_factory=list)
    cue_fields: list[str] = field(default_factory=list)
    stage_d_trace_available: bool = False
    evidence_attribution_backend: str = "gate_attention_score_proxy"
    output_contract_version: str = "stage_e_structured_output_v1"


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
