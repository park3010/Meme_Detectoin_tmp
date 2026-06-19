"""Dataclass schemas for Stage D."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from module.stage_a.schemas import StageAOutput
from module.stage_c.schemas import StageCOutput


@dataclass
class StageDInput:
    """Input container for Stage D."""

    stage_a: StageAOutput
    stage_c: StageCOutput


@dataclass
class StageDMetadata:
    """Auditable evidence-fusion and task-reasoning metadata."""

    internal_token_count: int
    verified_knowledge_count: int
    hidden_dim: int
    evidence_ids: list[str] = field(default_factory=list)
    knowledge_ids: list[str] = field(default_factory=list)
    regularizer_hooks: dict[str, float] = field(default_factory=dict)
    knowledge_need: float = 0.0
    support_matrix_shape: list[int] = field(default_factory=list)
    gate_mode: str = "token_sample_task_head"
    task_support_used: bool = False
    knowledge_origin_counts: dict[str, int] = field(default_factory=dict)
    knowledge_provenance_records: list[dict[str, Any]] = field(default_factory=list)
    attention_trace: dict[str, Any] = field(default_factory=dict)
    gate_statistics: dict[str, float] = field(default_factory=dict)
    task_support_summary: dict[str, float] = field(default_factory=dict)
    support_matrix_columns: list[str] = field(default_factory=list)
    stage_c_policy: dict[str, Any] = field(default_factory=dict)
    analysis_hooks: dict[str, float] = field(default_factory=dict)
    regularizer_hook_mode: str = "detached_analysis_only"
    regularizer_hooks_are_differentiable: bool = False


@dataclass
class StageDOutput:
    """Output from Stage D."""

    sample_id: str
    dataset_name: str
    shared_reasoning_state: torch.Tensor
    internal_memory: torch.Tensor
    fused_tokens: torch.Tensor
    cross_attention_weights: torch.Tensor
    gates: dict[str, Any]
    task_latents: dict[str, torch.Tensor]
    metadata: StageDMetadata
