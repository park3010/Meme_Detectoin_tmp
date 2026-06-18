"""Dataclass schemas for Stage C."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from module.stage_a.schemas import StageAOutput
from module.stage_b.schemas import KnowledgeCandidate, StageBOutput


@dataclass
class VerifiedKnowledgeItem:
    """A candidate that survived relevance and validation filtering."""

    knowledge_id: str
    text: str
    source: str
    relevance_score: float
    support_label: str
    support_score: float
    validity_score: float
    final_score: float
    token_index: int = -1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageCInput:
    """Input container for Stage C."""

    stage_a: StageAOutput
    stage_b: StageBOutput
    candidates: list[KnowledgeCandidate] | None = None


@dataclass
class StageCMetadata:
    """Stage C metadata."""

    input_candidate_count: int
    filtered_candidate_count: int
    top_k: int
    min_relevance: float
    claim_types: list[str] = field(default_factory=list)
    score_fields: list[str] = field(default_factory=list)
    support_matrix_columns: list[str] = field(default_factory=list)
    allow_low_relevance_fallback: bool = True


@dataclass
class StageCOutput:
    """Output from Stage C."""

    sample_id: str
    dataset_name: str
    verified_items: list[VerifiedKnowledgeItem]
    verified_tokens: torch.Tensor
    support_matrix: torch.Tensor
    final_scores: torch.Tensor
    internal_summary: str
    metadata: StageCMetadata

    @property
    def task_support_matrix(self) -> torch.Tensor:
        """Return target/intent/tactic support columns from the 6-column matrix."""

        if self.support_matrix.numel() == 0:
            return torch.zeros(0, 3, dtype=self.support_matrix.dtype)
        return self.support_matrix[:, 1:4]

    @property
    def named_support_columns(self) -> dict[str, torch.Tensor]:
        """Return support matrix columns keyed by semantic name."""

        columns = self.metadata.support_matrix_columns or [
            "relevance",
            "target_support",
            "intent_support",
            "tactic_support",
            "validity",
            "final",
        ]
        return {name: self.support_matrix[:, idx] for idx, name in enumerate(columns) if idx < self.support_matrix.size(1)}
