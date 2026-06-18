"""Dataclass schemas for Stage B."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from module.stage_a.schemas import StageAOutput


@dataclass
class QueryBundle:
    """Multiple query views for external knowledge acquisition."""

    ocr_query: str
    entity_queries: list[str] = field(default_factory=list)
    event_queries: list[str] = field(default_factory=list)
    meme_template_queries: list[str] = field(default_factory=list)
    social_context_queries: list[str] = field(default_factory=list)
    target_hypothesis_queries: list[str] = field(default_factory=list)

    def all_queries(self) -> list[str]:
        """Return unique non-empty queries in priority order."""

        queries = [
            self.ocr_query,
            *self.entity_queries,
            *self.event_queries,
            *self.meme_template_queries,
            *self.social_context_queries,
            *self.target_hypothesis_queries,
        ]
        seen: set[str] = set()
        result: list[str] = []
        for query in queries:
            clean = " ".join(str(query).split())
            if clean and clean not in seen:
                seen.add(clean)
                result.append(clean)
        return result


@dataclass
class LinkedEntity:
    """A lightweight entity or concept link."""

    surface: str
    normalized: str
    link_type: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeCandidate:
    """One retrieved or generated knowledge candidate."""

    candidate_id: str
    text: str
    source: str
    score: float
    candidate_type: str
    query: str = ""
    token_index: int = -1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageBInput:
    """Input container for external knowledge acquisition."""

    sample_id: str
    dataset_name: str
    ocr_text_full: str
    stage_a: StageAOutput
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageBMetadata:
    """Stage B metadata."""

    query_count: int
    linked_entity_count: int
    retrieved_count: int
    generated_count: int
    retriever_backend: str = "local"
    retrieval_stats: dict[str, Any] = field(default_factory=dict)
    query_types: dict[str, int] = field(default_factory=dict)
    evidence_surface_count: int = 0
    visual_evidence_used: bool = False
    fallback_candidates_used: bool = False
    query_source_breakdown: dict[str, int] = field(default_factory=dict)


@dataclass
class StageBOutput:
    """Output from Stage B."""

    sample_id: str
    dataset_name: str
    query_bundle: QueryBundle
    linked_entities: list[LinkedEntity]
    knowledge_candidates: list[KnowledgeCandidate]
    candidate_tokens: torch.Tensor
    generated_hypotheses: list[str]
    metadata: StageBMetadata
