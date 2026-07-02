"""Stage B: external knowledge acquisition."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from module.backbone.generation import GeneratorAdapter
from module.backbone.retrieval import CrossEncoderAdapter, LocalRetrieverAdapter
from module.backbone.text import TextEncoderWrapper
from module.internal_evidence_extractor import StageAOutput
from utils.tensor_utils import hashed_vector
from utils.text_utils import capitalized_spans, keyword_candidates, normalize_text, rhetorical_cues, sentence_chunks


# =============================================================================
# Stage B schemas
# =============================================================================

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
    """Candidate-state metadata produced before Stage C verification."""

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
    surface_records: list[dict[str, Any]] = field(default_factory=list)
    query_records: list[dict[str, Any]] = field(default_factory=list)


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


# =============================================================================
# Query construction and surface provenance
# =============================================================================

class QueryConstructor:
    """Build diverse external-knowledge queries from OCR and internal cues."""

    def __init__(self, max_queries: int = 6) -> None:
        self.max_queries = max_queries

    def build(self, ocr_text: str, stage_a: StageAOutput) -> QueryBundle:
        """Create query bundle for one sample."""

        clean = normalize_text(ocr_text)
        keywords = keyword_candidates(clean, limit=8)
        entities = capitalized_spans(ocr_text, limit=4)
        chunks = sentence_chunks(clean, limit=3)
        cues = rhetorical_cues(clean)
        aux_labels = stage_a.metadata.auxiliary_labels if hasattr(stage_a.metadata, "auxiliary_labels") else {}
        relation = (
            aux_labels.get("stage_a_multimodal_relation")
            or aux_labels.get("multimodal_relation")
            or ""
        ) if isinstance(aux_labels, dict) else ""
        target_like = [item.text for item in stage_a.evidence_items if item.evidence_type in {"text_span", "local_symbol"} and item.text]
        joined_keywords = " ".join(keywords[:6])
        return QueryBundle(
            ocr_query=clean[:280] if clean else joined_keywords,
            entity_queries=[f"{entity} identity context meme reference alias" for entity in entities],
            event_queries=[f"event background timeline {chunk}" for chunk in chunks if any(token in chunk.lower() for token in ["virus", "election", "debate", "brexit", "covid", "war", "school"])],
            meme_template_queries=[f"meme template rhetoric {relation} {' '.join(cues.keys())} {joined_keywords}".strip()]
            if joined_keywords or cues
            else [],
            social_context_queries=[f"social cultural context target group {joined_keywords}"] if joined_keywords else [],
            target_hypothesis_queries=[f"target intent tactic hypothesis {normalize_text(text)[:180]}" for text in target_like[:3]],
        )


# =============================================================================
# Entity/concept linking
# =============================================================================

class EntityConceptLinker:
    """Link surface entities/concepts and attach alias expansion metadata."""

    alias_table = {
        "usa": ["united states", "america", "american"],
        "america": ["united states", "usa", "american"],
        "covid": ["coronavirus", "covid-19", "pandemic"],
        "corona": ["coronavirus", "covid-19"],
        "trump": ["donald trump", "us president"],
        "obama": ["barack obama", "us president"],
        "brexit": ["uk european union withdrawal", "britain eu"],
    }

    def link(self, text: str, surface_forms: list[str] | None = None) -> list[LinkedEntity]:
        """Return entity-like and concept-like links from text plus evidence surfaces."""

        links: list[LinkedEntity] = []
        seen: set[str] = set()
        for span in capitalized_spans(text, limit=8):
            norm = span.lower()
            if norm in seen:
                continue
            seen.add(norm)
            links.append(
                LinkedEntity(
                    surface=span,
                    normalized=norm,
                    link_type="entity",
                    confidence=0.72,
                    metadata={"aliases": self.alias_table.get(norm, [])},
                )
            )
        for surface in surface_forms or []:
            clean = normalize_text(surface)
            norm = clean.lower()
            if not clean or norm in seen:
                continue
            seen.add(norm)
            link_type = "visual_symbol" if any(prefix in norm for prefix in ["region_", "patch", "roi"]) else "evidence_surface"
            confidence = 0.62 if link_type == "visual_symbol" else 0.56
            links.append(
                LinkedEntity(
                    surface=clean,
                    normalized=norm,
                    link_type=link_type,
                    confidence=confidence,
                    metadata={"aliases": self.alias_table.get(norm, []), "from_stage_a": True},
                )
            )
        for keyword in keyword_candidates(text, limit=10):
            if keyword in seen:
                continue
            seen.add(keyword)
            links.append(
                LinkedEntity(
                    surface=keyword,
                    normalized=keyword,
                    link_type="concept",
                    confidence=0.48,
                    metadata={"aliases": self.alias_table.get(keyword, [])},
                )
            )
        return links


# =============================================================================
# Hybrid retrieval and context augmentation
# =============================================================================

class HybridRetriever:
    """Run sparse+dense retrieval, rank fusion, and optional reranking."""

    def __init__(
        self,
        corpus_paths: list[str] | None = None,
        fallback_candidates: bool = True,
        top_k: int = 8,
        max_documents: int | None = None,
        use_cross_encoder_rerank: bool = True,
    ) -> None:
        self.adapter = LocalRetrieverAdapter(corpus_paths=corpus_paths, fallback_candidates=fallback_candidates, max_documents=max_documents)
        self.top_k = top_k
        self.cross_encoder = CrossEncoderAdapter() if use_cross_encoder_rerank else None

    def retrieve(self, query_bundle: QueryBundle) -> list[KnowledgeCandidate]:
        """Retrieve and deduplicate candidates across query types."""

        candidates: list[KnowledgeCandidate] = []
        seen_texts: set[str] = set()
        for query in query_bundle.all_queries():
            per_query_k = max(2, self.top_k // 2)
            for document in self.adapter.search(query, top_k=per_query_k):
                key = document.text.lower()[:240]
                if key in seen_texts:
                    continue
                seen_texts.add(key)
                score = float(document.metadata.get("retrieval_score", 0.0))
                rerank_score = self.cross_encoder.score(query, document.text) if self.cross_encoder else score
                fused_score = max(0.0, min(1.0, 0.6 * score + 0.4 * rerank_score))
                metadata = dict(document.metadata)
                metadata["cross_encoder_score"] = rerank_score
                metadata["query_type"] = _query_type(query_bundle, query)
                candidates.append(
                    KnowledgeCandidate(
                        candidate_id=document.doc_id,
                        text=document.text,
                        source=document.source,
                        score=fused_score,
                        candidate_type="retrieved",
                        query=query,
                        metadata=metadata,
                    )
                )
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[: self.top_k]


def _query_type(bundle: QueryBundle, query: str) -> str:
    if query == bundle.ocr_query:
        return "ocr"
    fields = {
        "entity": bundle.entity_queries,
        "event": bundle.event_queries,
        "meme_template": bundle.meme_template_queries,
        "social_context": bundle.social_context_queries,
        "target_hypothesis": bundle.target_hypothesis_queries,
    }
    for name, queries in fields.items():
        if query in queries:
            return name
    return "unknown"

class ContextAugmentationGenerator:
    """Generate short candidate hypotheses from top evidence."""

    def __init__(self, max_items: int = 3) -> None:
        self.generator = GeneratorAdapter()
        self.max_items = max_items

    def generate(self, ocr_text: str, retrieved: list[KnowledgeCandidate], sample_id: str = "sample") -> tuple[list[str], list[KnowledgeCandidate]]:
        """Return hypothesis strings grounded in retrieved evidence."""

        evidence_summary = "; ".join(normalize_text(item.text) for item in retrieved[:3])
        hypotheses = self.generator.generate_hypotheses(ocr_text, evidence_summary=evidence_summary, max_items=self.max_items)
        candidates = [
            KnowledgeCandidate(
                candidate_id=f"{sample_id}:generated_hypothesis:{idx}",
                text=hypothesis,
                source="template_generator",
                score=max(0.2, 0.55 - idx * 0.08),
                candidate_type="generated_hypothesis",
                metadata={"evidence_summary": evidence_summary[:500], "grounded_in": [item.candidate_id for item in retrieved[:3]]},
            )
            for idx, hypothesis in enumerate(hypotheses)
        ]
        return hypotheses, candidates


# =============================================================================
# Candidate provenance helpers
# =============================================================================



# =============================================================================
# ExternalKnowledgeAcquisition
# =============================================================================

class ExternalKnowledgeAcquisition(nn.Module):
    """Construct queries, link concepts, retrieve candidates, and generate hypotheses."""

    def __init__(
        self,
        hidden_dim: int = 256,
        corpus_paths: list[str] | None = None,
        top_k: int = 8,
        fallback_candidates: bool = True,
        prefer_transformers: bool = False,
        text_model_name: str = "microsoft/deberta-v3-base",
        text_checkpoint_path: str | Path | None = None,
        text_cache_dir: str | Path | None = None,
        text_local_files_only: bool = True,
        text_allow_download: bool = False,
        max_documents: int | None = None,
        use_cross_encoder_rerank: bool = True,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query_constructor = QueryConstructor(max_queries=top_k)
        self.linker = EntityConceptLinker()
        self.retriever = HybridRetriever(
            corpus_paths=corpus_paths,
            fallback_candidates=fallback_candidates,
            top_k=top_k,
            max_documents=max_documents,
            use_cross_encoder_rerank=use_cross_encoder_rerank,
        )
        self.generator = ContextAugmentationGenerator(max_items=3)
        self.text_encoder = TextEncoderWrapper(
            hidden_dim=hidden_dim,
            prefer_transformers=prefer_transformers,
            model_name=text_model_name,
            device=device,
            checkpoint_path=text_checkpoint_path,
            cache_dir=text_cache_dir,
            local_files_only=text_local_files_only,
            allow_download=text_allow_download,
        )
        self.source_embedding = nn.Embedding(6, hidden_dim)
        self.query_type_embedding = nn.Embedding(8, hidden_dim)
        self.score_projection = nn.Linear(3, hidden_dim)
        self.top_k = top_k

    def forward(self, stage_a: StageAOutput, ocr_text_full: str | None = None) -> StageBOutput:
        """Run Stage B from a Stage A output."""

        ocr_text = ocr_text_full or _extract_ocr_text(stage_a)
        bundle = self.query_constructor.build(ocr_text, stage_a)
        surface_records, surface_stats = collect_linkable_surface_records(stage_a, ocr_text)
        evidence_surfaces = _unique_surfaces([record["surface"] for record in surface_records])
        combined_link_text = " ".join([ocr_text, *evidence_surfaces])
        linked_entities = self.linker.link(combined_link_text, surface_forms=evidence_surfaces)
        _augment_queries_with_aliases(bundle, linked_entities)
        _augment_queries_with_evidence(bundle, stage_a, evidence_surfaces, linked_entities)
        query_records = _query_records(bundle, surface_records, linked_entities)
        retrieved = self.retriever.retrieve(bundle)
        hypotheses, generated = self.generator.generate(ocr_text, retrieved, sample_id=stage_a.sample_id)

        candidates = _dedupe_candidates([*retrieved, *generated])[: self.top_k]
        token_rows = []
        for idx, candidate in enumerate(candidates):
            _with_candidate_provenance(candidate)
            candidate.token_index = idx
            embedding, _, _ = self.text_encoder.encode(candidate.text)
            token_context = self._candidate_context_embedding(candidate)
            weighted = F.normalize(embedding + token_context, dim=0)
            token_rows.append(weighted)
        device = self.source_embedding.weight.device
        candidate_tokens = torch.stack(token_rows, dim=0) if token_rows else torch.zeros(0, self.hidden_dim, device=device)
        metadata = StageBMetadata(
            query_count=len(bundle.all_queries()),
            linked_entity_count=len(linked_entities),
            retrieved_count=len(retrieved),
            generated_count=len(generated),
            retriever_backend="local",
            retrieval_stats={
                "corpus_size": len(self.retriever.adapter.documents),
                "top_k": self.top_k,
                "has_fallback_candidates": any(candidate.source == "fallback" for candidate in candidates),
                "fallback_candidate_count": sum(1 for candidate in candidates if candidate.source == "fallback"),
            },
            query_types={
                "ocr": 1 if bundle.ocr_query else 0,
                "entity": len(bundle.entity_queries),
                "event": len(bundle.event_queries),
                "meme_template": len(bundle.meme_template_queries),
                "social_context": len(bundle.social_context_queries),
                "target_hypothesis": len(bundle.target_hypothesis_queries),
            },
            evidence_surface_count=len(evidence_surfaces),
            visual_evidence_used=surface_stats["visual_evidence_used"] > 0,
            fallback_candidates_used=any(candidate.source == "fallback" for candidate in candidates),
            query_source_breakdown=_query_source_breakdown(bundle),
            surface_records=surface_records,
            query_records=query_records,
        )
        return StageBOutput(
            sample_id=stage_a.sample_id,
            dataset_name=stage_a.dataset_name,
            query_bundle=bundle,
            linked_entities=linked_entities,
            knowledge_candidates=candidates,
            candidate_tokens=candidate_tokens,
            generated_hypotheses=hypotheses,
            metadata=metadata,
        )

    def _candidate_context_embedding(self, candidate: KnowledgeCandidate) -> torch.Tensor:
        """Encode source, query type, and score signals without changing token size."""

        source_id = _source_id(candidate.source)
        query_id = _query_type_id(str(candidate.metadata.get("query_type", "")))
        device = self.source_embedding.weight.device
        score_features = torch.tensor(
            [
                float(candidate.score),
                float(candidate.metadata.get("retrieval_score", candidate.score)),
                float(candidate.metadata.get("cross_encoder_score", 0.0)),
            ],
            dtype=torch.float32,
            device=device,
        )
        context = (
            self.source_embedding(torch.tensor(source_id, device=device))
            + self.query_type_embedding(torch.tensor(query_id, device=device))
            + self.score_projection(score_features)
        )
        return 0.15 * F.normalize(context, dim=0)


def _extract_ocr_text(stage_a: StageAOutput) -> str:
    for item in stage_a.evidence_items:
        if item.evidence_type == "global_text":
            return item.text
    return ""


def _dedupe_candidates(candidates: list[KnowledgeCandidate]) -> list[KnowledgeCandidate]:
    seen: set[str] = set()
    output: list[KnowledgeCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        key = candidate.text.lower()[:220]
        if key in seen:
            continue
        seen.add(key)
        output.append(candidate)
    return output


def _augment_queries_with_aliases(bundle: QueryBundle, linked_entities: list[LinkedEntity]) -> None:
    for entity in linked_entities:
        aliases = entity.metadata.get("aliases", []) if isinstance(entity.metadata, dict) else []
        for alias in aliases[:3]:
            query = f"{alias} {entity.link_type} background meme context"
            if query not in bundle.entity_queries:
                bundle.entity_queries.append(query)


def collect_linkable_surface_forms(stage_a: StageAOutput, ocr_text: str = "") -> tuple[list[str], dict[str, int]]:
    """Collect evidence-aware surfaces for linking and query augmentation."""

    records, stats = collect_linkable_surface_records(stage_a, ocr_text)
    return _unique_surfaces([record["surface"] for record in records]), stats


def collect_linkable_surface_records(stage_a: StageAOutput, ocr_text: str = "") -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Collect auditable linking surfaces with Stage A evidence provenance."""

    records: list[dict[str, Any]] = []
    stats = {
        "ocr": 0,
        "text_span": 0,
        "local_symbol": 0,
        "cross_modal_incongruity": 0,
        "metadata_keyword": 0,
        "roi_label": 0,
        "auxiliary_label": 0,
        "rhetorical_cue": 0,
        "visual_evidence_used": 0,
    }
    if ocr_text.strip():
        records.append(_surface_record(ocr_text, "ocr", source_stage="input", is_heuristic=False))
        stats["ocr"] += 1
    for item in stage_a.evidence_items:
        if item.evidence_type in {"text_span", "local_symbol", "cross_modal_incongruity"} and item.text:
            records.append(
                _surface_record(
                    item.text,
                    item.evidence_type,
                    source_stage=str(item.metadata.get("source_stage", "stage_a")),
                    evidence_id=item.evidence_id,
                    evidence_type=item.evidence_type,
                    modality=item.metadata.get("modality"),
                    grounding_type=item.metadata.get("grounding_type"),
                    is_heuristic=bool(item.metadata.get("is_heuristic", item.evidence_type != "text_span")),
                )
            )
            stats[item.evidence_type] += 1
            if item.evidence_type == "cross_modal_incongruity":
                stats["auxiliary_label"] += 1
        if item.evidence_type in {"local_symbol", "visual_patch"}:
            stats["visual_evidence_used"] += 1
        metadata = item.metadata or {}
        for keyword in metadata.get("top_keywords", []) or metadata.get("keywords", []) or []:
            records.append(
                _surface_record(
                    str(keyword),
                    "metadata_keyword",
                    source_stage=str(metadata.get("source_stage", "stage_a")),
                    evidence_id=item.evidence_id,
                    evidence_type=item.evidence_type,
                    modality=metadata.get("modality"),
                    grounding_type=metadata.get("grounding_type"),
                    is_heuristic=bool(metadata.get("is_heuristic", False)),
                )
            )
            stats["metadata_keyword"] += 1
        label = metadata.get("label")
        if label:
            records.append(
                _surface_record(
                    str(label),
                    "roi_label",
                    source_stage=str(metadata.get("source_stage", "stage_a")),
                    evidence_id=item.evidence_id,
                    evidence_type=item.evidence_type,
                    modality=metadata.get("modality"),
                    grounding_type=metadata.get("grounding_type"),
                    is_heuristic=bool(metadata.get("is_heuristic", True)),
                )
            )
            stats["roi_label"] += 1
    aux_labels = stage_a.metadata.auxiliary_labels if hasattr(stage_a.metadata, "auxiliary_labels") else {}
    relation = _stage_a_relation(stage_a)
    if relation:
        records.append(
            _surface_record(
                f"multimodal relation {relation}",
                "auxiliary_label",
                source_stage="stage_a",
                modality="cross_modal",
                grounding_type="cue",
                is_heuristic=True,
            )
        )
        stats["auxiliary_label"] += 1
    cues = aux_labels.get("rhetorical_cues", {}) if isinstance(aux_labels, dict) else {}
    if isinstance(cues, dict):
        for cue in cues:
            records.append(
                _surface_record(
                    f"rhetorical cue {cue}",
                    "rhetorical_cue",
                    source_stage="stage_a",
                    modality="text",
                    grounding_type="cue",
                    is_heuristic=True,
                )
            )
            stats["rhetorical_cue"] += 1
            stats["auxiliary_label"] += 1
    return _unique_surface_records(records), stats


def _augment_queries_with_evidence(
    bundle: QueryBundle,
    stage_a: StageAOutput,
    surfaces: list[str],
    linked_entities: list[LinkedEntity],
) -> None:
    visual_terms = [
        entity.surface
        for entity in linked_entities
        if entity.link_type in {"visual_symbol", "evidence_surface"} and any(token in entity.surface.lower() for token in ["region", "roi", "patch", "symbol"])
    ]
    local_symbols = [item.text for item in stage_a.evidence_items if item.evidence_type == "local_symbol" and item.text]
    relation = _stage_a_relation(stage_a)
    cues = stage_a.metadata.auxiliary_labels.get("rhetorical_cues", {}) if hasattr(stage_a.metadata, "auxiliary_labels") else {}
    cue_terms = list(cues.keys()) if isinstance(cues, dict) else []
    for symbol in [*visual_terms, *local_symbols][:4]:
        query = f"visual symbol meme meaning {symbol} {relation}".strip()
        if query not in bundle.meme_template_queries:
            bundle.meme_template_queries.append(query)
    if cue_terms:
        cue_query = f"social context rhetorical tactic {' '.join(cue_terms)} {' '.join(surfaces[:6])}".strip()
        if cue_query not in bundle.social_context_queries:
            bundle.social_context_queries.append(cue_query)
    target_like = [surface for surface in surfaces if len(surface.split()) <= 5][:4]
    for surface in target_like:
        query = f"target evidence span context {surface}"
        if query not in bundle.target_hypothesis_queries:
            bundle.target_hypothesis_queries.append(query)


def _unique_surfaces(surfaces: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for surface in surfaces:
        clean = " ".join(str(surface).split())
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            output.append(clean)
    return output


def _source_id(source: str) -> int:
    source = source.lower()
    if source == "fallback":
        return 0
    if "wiki" in source or "wikipedia" in source:
        return 1
    if source == "template_generator":
        return 2
    if "rag" in source:
        return 3
    return 4


def _query_type_id(query_type: str) -> int:
    mapping = {
        "ocr": 0,
        "entity": 1,
        "event": 2,
        "meme_template": 3,
        "social_context": 4,
        "target_hypothesis": 5,
    }
    return mapping.get(query_type, 6)


def _query_source_breakdown(bundle: QueryBundle) -> dict[str, int]:
    return {
        "ocr": 1 if bundle.ocr_query else 0,
        "entity": len(bundle.entity_queries),
        "event": len(bundle.event_queries),
        "meme_template": len(bundle.meme_template_queries),
        "social_context": len(bundle.social_context_queries),
        "target_hypothesis": len(bundle.target_hypothesis_queries),
    }


def _candidate_origin(candidate: KnowledgeCandidate) -> str:
    if candidate.candidate_type == "generated_hypothesis" or candidate.source == "template_generator":
        return "generated_hypothesis"
    if candidate.source == "fallback" or bool(candidate.metadata.get("fallback")):
        return "fallback"
    return "retrieved"


def _with_candidate_provenance(candidate: KnowledgeCandidate) -> KnowledgeCandidate:
    """Attach stable candidate-state provenance without changing ranking."""

    origin = _candidate_origin(candidate)
    candidate.metadata.update(
        {
            "source_stage": "stage_b",
            "candidate_origin": origin,
            "requires_verification": True,
            "is_retrieved": origin == "retrieved",
            "is_fallback": origin == "fallback",
            "is_generated": origin == "generated_hypothesis",
            "is_external_knowledge": origin == "retrieved",
        }
    )
    if origin == "generated_hypothesis":
        candidate.metadata["is_interpretive_hypothesis"] = True
    return candidate


def _surface_record(
    surface: str,
    surface_type: str,
    *,
    source_stage: str,
    evidence_id: str | None = None,
    evidence_type: str | None = None,
    modality: Any = None,
    grounding_type: Any = None,
    is_heuristic: bool,
) -> dict[str, Any]:
    return {
        "surface": " ".join(str(surface).split()),
        "surface_type": surface_type,
        "source_stage": source_stage,
        "evidence_id": evidence_id,
        "evidence_type": evidence_type,
        "modality": modality,
        "grounding_type": grounding_type,
        "is_heuristic": bool(is_heuristic),
    }


def _unique_surface_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str | None]] = set()
    output: list[dict[str, Any]] = []
    for record in records:
        surface = str(record.get("surface", "")).strip()
        key = (surface.lower(), str(record.get("surface_type", "")), record.get("evidence_id"))
        if not surface or key in seen:
            continue
        seen.add(key)
        output.append(record)
    return output


def _query_records(
    bundle: QueryBundle,
    surface_records: list[dict[str, Any]],
    linked_entities: list[LinkedEntity],
) -> list[dict[str, Any]]:
    typed_queries = [
        ("ocr", [bundle.ocr_query]),
        ("entity", bundle.entity_queries),
        ("event", bundle.event_queries),
        ("meme_template", bundle.meme_template_queries),
        ("social_context", bundle.social_context_queries),
        ("target_hypothesis", bundle.target_hypothesis_queries),
    ]
    evidence_ids = sorted({str(record["evidence_id"]) for record in surface_records if record.get("evidence_id")})
    evidence_types = sorted({str(record["evidence_type"]) for record in surface_records if record.get("evidence_type")})
    cue_types = sorted(
        {
            str(record["surface_type"])
            for record in surface_records
            if record.get("surface_type") in {"auxiliary_label", "rhetorical_cue", "cross_modal_incongruity"}
        }
    )
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    for query_type, queries in typed_queries:
        for query in queries:
            clean = " ".join(str(query).split())
            if not clean or clean in seen:
                continue
            seen.add(clean)
            records.append(
                {
                    "query_id": f"q{len(records)}",
                    "query_type": query_type,
                    "query": clean,
                    "source_stage": "stage_b",
                    "surface_count": len(surface_records),
                    "linked_entity_count": len(linked_entities),
                    "stage_a_evidence_ids": evidence_ids,
                    "stage_a_evidence_types": evidence_types,
                    "cue_types": cue_types,
                }
            )
    return records


def _stage_a_relation(stage_a: StageAOutput) -> str:
    aux_labels = stage_a.metadata.auxiliary_labels if hasattr(stage_a.metadata, "auxiliary_labels") else {}
    if not isinstance(aux_labels, dict):
        return ""
    return str(
        aux_labels.get("stage_a_multimodal_relation")
        or aux_labels.get("multimodal_relation")
        or ""
    )


__all__ = [
    "QueryBundle",
    "LinkedEntity",
    "KnowledgeCandidate",
    "StageBInput",
    "StageBMetadata",
    "StageBOutput",
    "QueryConstructor",
    "EntityConceptLinker",
    "HybridRetriever",
    "ContextAugmentationGenerator",
    "ExternalKnowledgeAcquisition",
    "collect_linkable_surface_forms",
    "collect_linkable_surface_records",
]
