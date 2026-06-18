"""Hybrid retrieval facade for Stage B."""

from __future__ import annotations

from module.backbones.cross_encoder_adapter import CrossEncoderAdapter
from module.backbones.retriever_adapter import LocalRetrieverAdapter
from module.stage_b.schemas import KnowledgeCandidate, QueryBundle


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
