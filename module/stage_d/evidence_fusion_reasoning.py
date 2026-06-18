"""Stage D orchestration module."""

from __future__ import annotations

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
        knowledge_need = float(stage_a.auxiliary_scores.get("knowledge_need", 0.5)) if hasattr(stage_a, "auxiliary_scores") else 0.5
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
