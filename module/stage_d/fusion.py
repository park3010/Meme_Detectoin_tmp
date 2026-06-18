"""Canonical Stage D fusion implementation imports."""

from module.stage_d.evidence_fusion_reasoning import EvidenceFusionReasoning
from module.stage_d.gating import EvidenceGate
from module.stage_d.internal_aggregator import InternalEvidenceAggregator
from module.stage_d.knowledge_cross_attention import KnowledgeConditionedCrossAttention
from module.stage_d.task_reasoning import TaskAwareReasoningHeads

__all__ = [
    "InternalEvidenceAggregator",
    "KnowledgeConditionedCrossAttention",
    "EvidenceGate",
    "TaskAwareReasoningHeads",
    "EvidenceFusionReasoning",
]
