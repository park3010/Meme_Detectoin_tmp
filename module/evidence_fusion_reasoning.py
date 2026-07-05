"""Stage D: evidence fusion and reasoning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from module.internal_evidence_extractor import StageAOutput
from module.knowledge_filter_verifier import StageCOutput
from utils.tensor_utils import hashed_vector, stable_hash


# =============================================================================
# Stage D schemas
# =============================================================================

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


# =============================================================================
# Internal evidence aggregation
# =============================================================================

class InternalEvidenceAggregator(nn.Module):
    """Aggregate internal evidence with type and dataset embeddings."""

    def __init__(self, hidden_dim: int = 256, num_heads: int = 4, num_layers: int = 2, dataset_buckets: int = 32) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.type_to_id = {
            "global_visual": 0,
            "global_text": 1,
            "local_symbol": 2,
            "cross_modal_incongruity": 3,
            "text_span": 4,
            "visual_patch": 5,
        }
        self.type_embedding = nn.Embedding(8, hidden_dim)
        self.dataset_embedding = nn.Embedding(dataset_buckets, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dataset_buckets = dataset_buckets

    def forward(self, stage_a: StageAOutput) -> tuple[torch.Tensor, torch.Tensor]:
        """Return contextualized internal memory and a summary vector."""

        device = self.type_embedding.weight.device
        tokens = stage_a.internal_tokens.to(device=device, dtype=self.type_embedding.weight.dtype)
        if tokens.numel() == 0:
            tokens = torch.zeros(1, self.hidden_dim, device=device)
        type_ids = []
        for item in stage_a.evidence_items[: tokens.size(0)]:
            type_ids.append(self.type_to_id.get(item.evidence_type, 7))
        while len(type_ids) < tokens.size(0):
            type_ids.append(7)
        type_tensor = torch.tensor(type_ids, dtype=torch.long, device=tokens.device)
        dataset_id = stable_hash(stage_a.dataset_name, self.dataset_buckets)
        dataset_tensor = torch.full((tokens.size(0),), dataset_id, dtype=torch.long, device=tokens.device)
        enriched = tokens + self.type_embedding(type_tensor) + self.dataset_embedding(dataset_tensor)
        memory = self.encoder(enriched.unsqueeze(0)).squeeze(0)
        memory = self.norm(memory)
        summary = memory.mean(dim=0)
        return memory, summary


# =============================================================================
# Knowledge-conditioned cross-attention
# =============================================================================

class KnowledgeConditionedCrossAttention(nn.Module):
    """Fuse internal memory with verified knowledge tokens."""

    def __init__(self, hidden_dim: int = 256, num_heads: int = 4) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.0, batch_first=True)
        self.support_projection = nn.Linear(1, hidden_dim)
        self.task_support_projection = nn.Linear(4, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, hidden_dim))

    def forward(
        self,
        internal_memory: torch.Tensor,
        knowledge_tokens: torch.Tensor,
        support_scores: torch.Tensor | None = None,
        task_support_matrix: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fused internal tokens and attention weights."""

        if knowledge_tokens.numel() == 0 or knowledge_tokens.size(0) == 0:
            weights = torch.zeros(internal_memory.size(0), 0, dtype=internal_memory.dtype, device=internal_memory.device)
            return internal_memory, weights
        query = internal_memory.unsqueeze(0)
        key_value_tokens = knowledge_tokens.to(device=internal_memory.device, dtype=internal_memory.dtype)
        if support_scores is not None and support_scores.numel() == key_value_tokens.size(0):
            support = support_scores.to(device=internal_memory.device, dtype=internal_memory.dtype).clamp(0.0, 1.0).unsqueeze(-1)
            if task_support_matrix is not None and task_support_matrix.numel() and task_support_matrix.size(0) == key_value_tokens.size(0):
                task_support = task_support_matrix.to(device=internal_memory.device, dtype=internal_memory.dtype).clamp(0.0, 1.0)
                key_value_tokens = key_value_tokens + self.task_support_projection(torch.cat([support, task_support], dim=-1))
            else:
                key_value_tokens = key_value_tokens + self.support_projection(support)
        key_value = key_value_tokens.unsqueeze(0)
        attended, weights = self.attention(query=query, key=key_value, value=key_value, need_weights=True)
        fused = self.norm(internal_memory + attended.squeeze(0))
        fused = self.norm(fused + self.ffn(fused))
        return fused, weights.squeeze(0)


# =============================================================================
# Evidence gating
# =============================================================================

class EvidenceGate(nn.Module):
    """Compute token-, sample-, and task-level gates."""

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.token_gate = nn.Linear(hidden_dim, 1)
        self.compare_token_gate = nn.Linear(hidden_dim * 4, 1)
        self.sample_gate = nn.Linear(hidden_dim, 1)
        self.knowledge_need_gate = nn.Linear(1, 1)
        self.task_gate = nn.Linear(hidden_dim, 3)
        self.head_gate = nn.Linear(hidden_dim, 3)

    def forward(
        self,
        fused_tokens: torch.Tensor,
        internal_memory: torch.Tensor | None = None,
        knowledge_need: float | torch.Tensor | None = None,
        q_int: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Apply gates and return gated tokens plus gate tensors."""

        if internal_memory is None:
            internal_memory = fused_tokens
        if internal_memory.shape != fused_tokens.shape:
            raise ValueError(f"internal_memory shape {tuple(internal_memory.shape)} must match fused_tokens {tuple(fused_tokens.shape)}")

        compare = torch.cat([internal_memory, fused_tokens, torch.abs(fused_tokens - internal_memory), fused_tokens * internal_memory], dim=-1)
        token_level = torch.sigmoid(0.5 * self.token_gate(fused_tokens).squeeze(-1) + 0.5 * self.compare_token_gate(compare).squeeze(-1))
        summary = q_int if q_int is not None else fused_tokens.mean(dim=0)
        learned_sample = torch.sigmoid(self.sample_gate(summary)).squeeze()
        need_tensor = _knowledge_need_tensor(knowledge_need, fused_tokens.device, fused_tokens.dtype)
        need_gate = torch.sigmoid(self.knowledge_need_gate(need_tensor.view(1))).squeeze()
        sample_level = (0.65 * learned_sample + 0.35 * need_tensor.squeeze()).clamp(0.0, 1.0) * (0.5 + 0.5 * need_gate)
        task_level = torch.sigmoid(self.task_gate(summary))
        head_level = torch.sigmoid(self.head_gate(fused_tokens)).transpose(0, 1)
        external_gate = (token_level * sample_level).clamp(0.0, 1.0)
        gated = external_gate.unsqueeze(-1) * fused_tokens + (1.0 - external_gate).unsqueeze(-1) * internal_memory
        return gated, {
            "token_level": token_level,
            "sample_level": sample_level,
            "knowledge_need": need_tensor.squeeze(),
            "external_gate": external_gate,
            "task_level": task_level,
            "head_level": head_level,
        }


def _knowledge_need_tensor(value: float | torch.Tensor | None, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype).flatten()[0].clamp(0.0, 1.0)
    if value is None:
        return torch.tensor(0.5, device=device, dtype=dtype)
    return torch.tensor(float(value), device=device, dtype=dtype).clamp(0.0, 1.0)


# =============================================================================
# Task-aware reasoning heads
# =============================================================================

class TaskAwareReasoningHeads(nn.Module):
    """Produce shared state and task-specific latent vectors."""

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.private_norm = nn.LayerNorm(hidden_dim)
        self.target = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.intent = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.tactic = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))

    def forward(
        self,
        gated_tokens: torch.Tensor,
        task_gates: torch.Tensor,
        head_level: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return shared reasoning state and task latents."""

        pooled = gated_tokens.mean(dim=0)
        shared_state = self.shared(pooled)
        gates = task_gates.flatten()
        private_pools = _private_pools(gated_tokens, head_level)
        latents = {
            "target": self.target(shared_state + self.private_norm(private_pools["target"])) * gates[0],
            "intent": self.intent(shared_state + self.private_norm(private_pools["intent"])) * gates[1],
            "tactic": self.tactic(shared_state + self.private_norm(private_pools["tactic"])) * gates[2],
        }
        return shared_state, latents


def _private_pools(gated_tokens: torch.Tensor, head_level: torch.Tensor | None) -> dict[str, torch.Tensor]:
    names = ["target", "intent", "tactic"]
    if head_level is None or head_level.numel() == 0:
        pooled = gated_tokens.mean(dim=0)
        return {name: pooled for name in names}
    pools: dict[str, torch.Tensor] = {}
    for idx, name in enumerate(names):
        weights = head_level[idx].float()
        weights = weights / weights.sum().clamp(min=1e-6)
        pools[name] = (gated_tokens * weights.unsqueeze(-1)).sum(dim=0)
    return pools


# =============================================================================
# Stage D diagnostics and trace helpers
# =============================================================================



# =============================================================================
# EvidenceFusionReasoning
# =============================================================================

class EvidenceFusionReasoning(nn.Module):
    """Fuse internal evidence and verified knowledge into task-aware reasoning states."""

    def __init__(self, hidden_dim: int = 256, num_heads: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.internal_aggregator = InternalEvidenceAggregator(hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers)
        self.cross_attention = KnowledgeConditionedCrossAttention(hidden_dim=hidden_dim, num_heads=num_heads)
        self.gate = EvidenceGate(hidden_dim=hidden_dim)
        self.reasoning = TaskAwareReasoningHeads(hidden_dim=hidden_dim)

    def forward(self, stage_a: StageAOutput, stage_c: StageCOutput, *, disable_task_aware_gate: bool = False) -> StageDOutput:
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
        gate_mode = "token_gated_internal_external_mix"
        if disable_task_aware_gate:
            shared_gate = gates["task_level"].mean().repeat(3)
            gates["task_level"] = shared_gate
            gate_mode = "shared_gate_train_time_ablation"
        shared_state, task_latents = self.reasoning(gated_tokens, gates["task_level"], head_level=gates.get("head_level"))
        regularizers = _regularizer_hooks(gates, attention_weights, stage_c.final_scores)
        provenance_records = _knowledge_provenance_records(stage_c)
        gate_statistics = _gate_statistics(gates)
        gate_statistics["knowledge_need"] = knowledge_need
        task_support_summary = _task_support_summary(stage_c)
        analysis_hooks = dict(regularizers)
        if disable_task_aware_gate:
            analysis_hooks["task_aware_gate_enabled"] = 0.0
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
                gate_mode=gate_mode,
                task_support_used=task_support_matrix.numel() > 0,
                knowledge_origin_counts=_knowledge_origin_counts(provenance_records),
                knowledge_provenance_records=provenance_records,
                attention_trace=_attention_trace(attention_weights, stage_a, stage_c),
                gate_statistics=gate_statistics,
                task_support_summary=task_support_summary,
                support_matrix_columns=list(getattr(stage_c.metadata, "support_matrix_columns", []) or []),
                stage_c_policy=_stage_c_policy(stage_c),
                analysis_hooks=analysis_hooks,
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


__all__ = [
    "StageDInput",
    "StageDMetadata",
    "StageDOutput",
    "InternalEvidenceAggregator",
    "KnowledgeConditionedCrossAttention",
    "EvidenceGate",
    "TaskAwareReasoningHeads",
    "EvidenceFusionReasoning",
]
