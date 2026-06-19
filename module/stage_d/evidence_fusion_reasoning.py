"""Stage D orchestration module."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from module.stage_a.schemas import StageAOutput
from module.stage_c.schemas import StageCOutput
from module.stage_d.gating import EvidenceGate
from module.stage_d.internal_aggregator import InternalEvidenceAggregator
from module.stage_d.knowledge_cross_attention import KnowledgeConditionedCrossAttention
from module.stage_d.schemas import StageDMetadata, StageDOutput
from module.stage_d.task_reasoning import TaskAwareReasoningHeads


class EvidenceFusionReasoning(nn.Module):
    """Fuse internal evidence and verified knowledge into task-aware reasoning states."""

    def __init__(self, hidden_dim: int = 256, num_heads: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.internal_aggregator = InternalEvidenceAggregator(hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers)
        self.cross_attention = KnowledgeConditionedCrossAttention(hidden_dim=hidden_dim, num_heads=num_heads)
        self.gate = EvidenceGate(hidden_dim=hidden_dim)
        self.reasoning = TaskAwareReasoningHeads(hidden_dim=hidden_dim)

    def forward(self, stage_a: StageAOutput, stage_c: StageCOutput) -> StageDOutput:
        """Run Stage D for one sample."""

        internal_memory, _ = self.internal_aggregator(stage_a)
        knowledge_need = (
            float(
                stage_a.auxiliary_scores.get(
                    "knowledge_need_score",
                    stage_a.auxiliary_scores.get("knowledge_need", 0.5),
                )
            )
            if hasattr(stage_a, "auxiliary_scores")
            else 0.5
        )
        task_support_matrix = _task_support_matrix(stage_c)
        fused_tokens, attention_weights = self.cross_attention(
            internal_memory,
            stage_c.verified_tokens,
            support_scores=stage_c.final_scores,
            task_support_matrix=task_support_matrix,
        )
        gated_tokens, gates = self.gate(
            fused_tokens,
            internal_memory=internal_memory,
            knowledge_need=knowledge_need,
            q_int=internal_memory.mean(dim=0),
        )
        shared_state, task_latents = self.reasoning(gated_tokens, gates["task_level"], head_level=gates.get("head_level"))
        regularizers = _regularizer_hooks(gates, attention_weights, stage_c.final_scores)
        provenance_records = _knowledge_provenance_records(stage_c)
        gate_statistics = _gate_statistics(gates)
        gate_statistics["knowledge_need"] = knowledge_need
        task_support_summary = _task_support_summary(stage_c)
        return StageDOutput(
            sample_id=stage_a.sample_id,
            dataset_name=stage_a.dataset_name,
            shared_reasoning_state=shared_state,
            internal_memory=internal_memory,
            fused_tokens=fused_tokens,
            cross_attention_weights=attention_weights,
            gates=gates,
            task_latents=task_latents,
            metadata=StageDMetadata(
                internal_token_count=stage_a.internal_tokens.size(0),
                verified_knowledge_count=stage_c.verified_tokens.size(0),
                hidden_dim=self.hidden_dim,
                evidence_ids=[item.evidence_id for item in stage_a.evidence_items],
                knowledge_ids=[item.knowledge_id for item in stage_c.verified_items],
                regularizer_hooks=regularizers,
                knowledge_need=knowledge_need,
                support_matrix_shape=list(stage_c.support_matrix.shape),
                gate_mode="token_gated_internal_external_mix",
                task_support_used=task_support_matrix.numel() > 0,
                knowledge_origin_counts=_knowledge_origin_counts(provenance_records),
                knowledge_provenance_records=provenance_records,
                attention_trace=_attention_trace(attention_weights, stage_a, stage_c),
                gate_statistics=gate_statistics,
                task_support_summary=task_support_summary,
                support_matrix_columns=list(getattr(stage_c.metadata, "support_matrix_columns", []) or []),
                stage_c_policy=_stage_c_policy(stage_c),
                analysis_hooks=dict(regularizers),
                regularizer_hook_mode="detached_analysis_only",
                regularizer_hooks_are_differentiable=False,
            ),
        )


def _regularizer_hooks(gates, attention_weights, final_scores) -> dict[str, float]:
    token_gate = gates.get("token_level")
    head_level = gates.get("head_level")
    hooks: dict[str, float] = {}
    if token_gate is not None and token_gate.numel() > 0:
        hooks["token_gate_sparsity"] = float(token_gate.detach().mean())
    if head_level is not None and head_level.numel() > 0:
        hooks["head_gate_diversity"] = float(head_level.detach().std())
    if attention_weights.numel() > 0:
        hooks["attention_entropy_proxy"] = float((attention_weights.detach() * (1.0 - attention_weights.detach())).mean())
    if final_scores.numel() > 0:
        hooks["knowledge_sufficiency"] = float(final_scores.detach().mean())
    return hooks


def _task_support_matrix(stage_c: StageCOutput):
    try:
        return stage_c.task_support_matrix
    except AttributeError:
        if stage_c.support_matrix.numel() and stage_c.support_matrix.size(1) >= 4:
            return stage_c.support_matrix[:, 1:4]
        if stage_c.final_scores.numel():
            return stage_c.final_scores.unsqueeze(-1).repeat(1, 3)
        return stage_c.support_matrix.new_zeros((0, 3))


def _knowledge_provenance_records(stage_c: StageCOutput) -> list[dict[str, Any]]:
    """Return compact provenance and verification scores for fused knowledge."""

    records: list[dict[str, Any]] = []
    for item in stage_c.verified_items:
        metadata = item.metadata or {}
        records.append(
            {
                "knowledge_id": item.knowledge_id,
                "source": item.source,
                "token_index": item.token_index,
                "candidate_origin": str(metadata.get("candidate_origin", "unknown")),
                "is_external_knowledge": bool(metadata.get("is_external_knowledge", False)),
                "is_generated": bool(metadata.get("is_generated", False)),
                "is_fallback": bool(metadata.get("is_fallback", False)),
                "is_retrieved": bool(metadata.get("is_retrieved", False)),
                "support_label": item.support_label,
                "support_score": float(item.support_score),
                "relevance_score": float(item.relevance_score),
                "validity_score": float(item.validity_score),
                "final_score": float(item.final_score),
            }
        )
    return records


def _knowledge_origin_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        origin = str(record.get("candidate_origin", "unknown"))
        counts[origin] = counts.get(origin, 0) + 1
    return counts


def _attention_trace(
    attention_weights: torch.Tensor,
    stage_a: StageAOutput,
    stage_c: StageCOutput,
    *,
    top_k: int = 8,
) -> dict[str, Any]:
    """Summarize strongest internal-evidence/knowledge attention links."""

    shape = list(attention_weights.shape)
    if attention_weights.numel() == 0 or not stage_c.verified_items:
        return {"top_links": [], "has_attention": False, "attention_shape": shape}
    weights = attention_weights.detach()
    while weights.dim() > 2 and weights.size(0) == 1:
        weights = weights.squeeze(0)
    expected_shape = (len(stage_a.evidence_items), len(stage_c.verified_items))
    if weights.dim() != 2 or tuple(weights.shape) != expected_shape:
        return {
            "top_links": [],
            "has_attention": False,
            "reason": "shape_mismatch",
            "attention_shape": shape,
            "expected_shape": list(expected_shape),
        }
    flat = weights.flatten()
    count = min(max(0, int(top_k)), flat.numel())
    if count == 0:
        return {"top_links": [], "has_attention": False, "attention_shape": shape}
    values, indices = torch.topk(flat, k=count)
    links: list[dict[str, Any]] = []
    knowledge_count = weights.size(1)
    for value, flat_index in zip(values, indices):
        evidence_index = int(flat_index) // knowledge_count
        knowledge_index = int(flat_index) % knowledge_count
        evidence = stage_a.evidence_items[evidence_index]
        knowledge = stage_c.verified_items[knowledge_index]
        links.append(
            {
                "evidence_id": evidence.evidence_id,
                "evidence_type": evidence.evidence_type,
                "knowledge_id": knowledge.knowledge_id,
                "candidate_origin": str(knowledge.metadata.get("candidate_origin", "unknown")),
                "attention_weight": float(value),
                "knowledge_final_score": float(knowledge.final_score),
                "knowledge_support_label": knowledge.support_label,
            }
        )
    return {"top_links": links, "has_attention": True, "attention_shape": shape}


def _gate_statistics(gates: dict[str, Any]) -> dict[str, float]:
    """Return detached scalar summaries without changing full gate tensors."""

    stats: dict[str, float] = {}
    _add_tensor_summary(stats, gates.get("token_level"), "token_level")
    _add_tensor_summary(stats, gates.get("external_gate"), "external_gate", include_range=True)
    sample_level = gates.get("sample_level")
    if isinstance(sample_level, torch.Tensor) and sample_level.numel():
        stats["sample_level"] = float(sample_level.detach().float().mean())
    knowledge_need = gates.get("knowledge_need")
    if isinstance(knowledge_need, torch.Tensor) and knowledge_need.numel():
        stats["knowledge_need"] = float(knowledge_need.detach().float().mean())
    task_level = gates.get("task_level")
    if isinstance(task_level, torch.Tensor) and task_level.numel() >= 3:
        values = task_level.detach().float().flatten()
        for index, name in enumerate(["target", "intent", "tactic"]):
            stats[f"task_gate_{name}"] = float(values[index])
    head_level = gates.get("head_level")
    if isinstance(head_level, torch.Tensor) and head_level.numel() and head_level.dim() >= 2:
        values = head_level.detach().float()
        for index, name in enumerate(["target", "intent", "tactic"]):
            if index < values.size(0):
                stats[f"head_gate_mean_{name}"] = float(values[index].mean())
    return stats


def _add_tensor_summary(
    output: dict[str, float],
    value: Any,
    prefix: str,
    *,
    include_range: bool = False,
) -> None:
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return
    tensor = value.detach().float()
    output[f"{prefix}_mean"] = float(tensor.mean())
    output[f"{prefix}_std"] = float(tensor.std(unbiased=False))
    if include_range:
        output[f"{prefix}_min"] = float(tensor.min())
        output[f"{prefix}_max"] = float(tensor.max())


def _task_support_summary(stage_c: StageCOutput) -> dict[str, float]:
    matrix = _task_support_matrix(stage_c)
    if matrix.numel() == 0 or matrix.dim() != 2 or matrix.size(1) < 3:
        return {
            "verified_knowledge_count": float(len(stage_c.verified_items)),
            "task_support_used": False,
        }
    matrix = matrix.detach().float()
    summary: dict[str, float] = {
        "verified_knowledge_count": float(len(stage_c.verified_items)),
        "task_support_used": True,
    }
    for index, name in enumerate(["target", "intent", "tactic"]):
        summary[f"{name}_support_mean"] = float(matrix[:, index].mean())
        summary[f"{name}_support_max"] = float(matrix[:, index].max())
    return summary


def _stage_c_policy(stage_c: StageCOutput) -> dict[str, Any]:
    metadata = stage_c.metadata
    return {
        "score_weights": dict(getattr(metadata, "score_weights", {}) or {}),
        "label_bonus": dict(getattr(metadata, "label_bonus", {}) or {}),
        "verification_policy": dict(getattr(metadata, "verification_policy", {}) or {}),
        "verified_origin_counts": dict(getattr(metadata, "verified_origin_counts", {}) or {}),
        "input_origin_counts": dict(getattr(metadata, "input_origin_counts", {}) or {}),
    }
