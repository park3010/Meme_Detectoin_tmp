"""Stage B orchestration module."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from module.backbones.text_encoder_wrapper import TextEncoderWrapper
from module.stage_a.schemas import StageAOutput
from module.stage_b.context_generator import ContextAugmentationGenerator
from module.stage_b.entity_linker import EntityConceptLinker
from module.stage_b.hybrid_retriever import HybridRetriever
from module.stage_b.query_constructor import QueryConstructor
from module.stage_b.schemas import KnowledgeCandidate, LinkedEntity, QueryBundle, StageBInput, StageBMetadata, StageBOutput


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
        self.text_encoder = TextEncoderWrapper(hidden_dim=hidden_dim, prefer_transformers=prefer_transformers, model_name=text_model_name, device=device)
        self.source_embedding = nn.Embedding(6, hidden_dim)
        self.query_type_embedding = nn.Embedding(8, hidden_dim)
        self.score_projection = nn.Linear(3, hidden_dim)
        self.top_k = top_k

    def forward(self, stage_a: StageAOutput, ocr_text_full: str | None = None) -> StageBOutput:
        """Run Stage B from a Stage A output."""

        ocr_text = ocr_text_full or _extract_ocr_text(stage_a)
        bundle = self.query_constructor.build(ocr_text, stage_a)
        evidence_surfaces, surface_stats = collect_linkable_surface_forms(stage_a, ocr_text)
        combined_link_text = " ".join([ocr_text, *evidence_surfaces])
        linked_entities = self.linker.link(combined_link_text, surface_forms=evidence_surfaces)
        _augment_queries_with_aliases(bundle, linked_entities)
        _augment_queries_with_evidence(bundle, stage_a, evidence_surfaces, linked_entities)
        retrieved = self.retriever.retrieve(bundle)
        hypotheses, generated = self.generator.generate(ocr_text, retrieved, sample_id=stage_a.sample_id)

        candidates = _dedupe_candidates([*retrieved, *generated])[: self.top_k]
        token_rows = []
        for idx, candidate in enumerate(candidates):
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

    surfaces: list[str] = []
    stats = {"ocr": 0, "text_span": 0, "local_symbol": 0, "metadata_keyword": 0, "roi_label": 0, "auxiliary_label": 0, "visual_evidence_used": 0}
    if ocr_text.strip():
        surfaces.append(ocr_text)
        stats["ocr"] += 1
    for item in stage_a.evidence_items:
        if item.evidence_type in {"text_span", "local_symbol", "cross_modal_incongruity"} and item.text:
            surfaces.append(item.text)
            stats[item.evidence_type if item.evidence_type in stats else "auxiliary_label"] = stats.get(item.evidence_type, 0) + 1
        if item.evidence_type in {"local_symbol", "visual_patch"}:
            stats["visual_evidence_used"] += 1
        metadata = item.metadata or {}
        for keyword in metadata.get("top_keywords", []) or metadata.get("keywords", []) or []:
            surfaces.append(str(keyword))
            stats["metadata_keyword"] += 1
        label = metadata.get("label")
        if label:
            surfaces.append(str(label))
            stats["roi_label"] += 1
    aux_labels = stage_a.metadata.auxiliary_labels if hasattr(stage_a.metadata, "auxiliary_labels") else {}
    relation = aux_labels.get("multimodal_relation") if isinstance(aux_labels, dict) else None
    if relation:
        surfaces.append(f"multimodal relation {relation}")
        stats["auxiliary_label"] += 1
    cues = aux_labels.get("rhetorical_cues", {}) if isinstance(aux_labels, dict) else {}
    if isinstance(cues, dict):
        for cue in cues:
            surfaces.append(f"rhetorical cue {cue}")
            stats["auxiliary_label"] += 1
    return _unique_surfaces(surfaces), stats


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
    relation = stage_a.metadata.auxiliary_labels.get("multimodal_relation", "") if hasattr(stage_a.metadata, "auxiliary_labels") else ""
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
