"""Canonical Stage C verifier implementation imports."""

from module.stage_c.knowledge_filter_verifier import KnowledgeRelevanceFilterVerifier
from module.stage_c.redundancy_reducer import RedundancyReducer
from module.stage_c.relevance_scorer import EvidenceAwareRelevanceScorer
from module.stage_c.support_verifier import SupportContradictionVerifier
from module.stage_c.validator import CredibilityTemporalValidator

__all__ = [
    "EvidenceAwareRelevanceScorer",
    "SupportContradictionVerifier",
    "CredibilityTemporalValidator",
    "RedundancyReducer",
    "KnowledgeRelevanceFilterVerifier",
]
