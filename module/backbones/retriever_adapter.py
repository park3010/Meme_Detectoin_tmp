"""Sparse/dense retrieval adapter with a tiny local backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from utils.io import read_jsonl
from utils.retrieval_utils import bm25_like_score, lexical_retrieval_score, reciprocal_rank_fusion
from utils.tensor_utils import hashed_vector
from utils.text_utils import keyword_candidates, normalize_text


@dataclass
class KnowledgeDocument:
    """A retrievable knowledge document."""

    doc_id: str
    text: str
    source: str = "local"
    metadata: dict[str, Any] = field(default_factory=dict)


class LocalRetrieverAdapter:
    """Minimal sparse+dense search over optional local corpora.

    The class is intentionally small but mirrors a future BM25/FAISS adapter:
    sparse and dense searches are exposed independently, and `search` performs
    rank fusion plus fallback candidate generation when no corpus is available.
    """

    def __init__(
        self,
        corpus_paths: list[str | Path] | None = None,
        fallback_candidates: bool = True,
        dense_dim: int = 256,
        max_documents: int | None = None,
    ) -> None:
        self.dense_dim = dense_dim
        self.documents = self._load_corpus(corpus_paths or [], max_documents=max_documents)
        self.avg_doc_len = sum(len(doc.text.split()) for doc in self.documents) / max(1, len(self.documents))
        self.document_embeddings = [hashed_vector(document.text, dim=dense_dim) for document in self.documents]
        self.fallback_candidates = fallback_candidates

    def search_sparse(self, query: str, top_k: int = 8) -> list[KnowledgeDocument]:
        """Return sparse lexical/BM25-style matches."""

        scored = [
            (0.45 * lexical_retrieval_score(query, document.text) + 0.55 * bm25_like_score(query, document.text, self.avg_doc_len), document)
            for document in self.documents
            if document.text.strip()
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [self._with_score(document, score, "sparse_score") for score, document in scored[:top_k] if score > 0]

    def search_dense(self, query: str, top_k: int = 8) -> list[KnowledgeDocument]:
        """Return hashed-vector dense matches as a FAISS-compatible fallback."""

        if not self.documents:
            return []
        query_vector = hashed_vector(query, dim=self.dense_dim)
        scored = [
            (float(F.cosine_similarity(query_vector, doc_vector, dim=0).clamp(min=0.0)), document)
            for doc_vector, document in zip(self.document_embeddings, self.documents)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [self._with_score(document, score, "dense_score") for score, document in scored[:top_k] if score > 0]

    def search(self, query: str, top_k: int = 8) -> list[KnowledgeDocument]:
        """Return hybrid sparse+dense ranked documents for the query."""

        sparse = self.search_sparse(query, top_k=max(top_k * 2, 4))
        dense = self.search_dense(query, top_k=max(top_k * 2, 4))
        by_id = {doc.doc_id: doc for doc in [*sparse, *dense]}
        rankings = [
            [(doc.doc_id, float(doc.metadata.get("sparse_score", 0.0))) for doc in sparse],
            [(doc.doc_id, float(doc.metadata.get("dense_score", 0.0))) for doc in dense],
        ]
        fused = reciprocal_rank_fusion(rankings)
        results: list[KnowledgeDocument] = []
        for doc_id, fusion_score in sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]:
            document = by_id[doc_id]
            metadata = dict(document.metadata)
            sparse_score = metadata.get("sparse_score", 0.0)
            dense_score = metadata.get("dense_score", 0.0)
            metadata["fusion_score"] = fusion_score
            metadata["retrieval_score"] = max(float(sparse_score), float(dense_score), float(fusion_score))
            results.append(KnowledgeDocument(document.doc_id, document.text, document.source, metadata))
        if results or not self.fallback_candidates:
            return results
        return self._fallback_results(query, top_k)

    def _load_corpus(self, corpus_paths: list[str | Path], max_documents: int | None = None) -> list[KnowledgeDocument]:
        documents: list[KnowledgeDocument] = []
        for path in self._expand_paths(corpus_paths):
            if not path.exists():
                continue
            if path.suffix.lower() == ".jsonl":
                records = read_jsonl(path)
                for idx, record in enumerate(records):
                    text = normalize_text(
                        record.get("text")
                        or record.get("contents")
                        or record.get("content")
                        or record.get("passage")
                        or record.get("summary")
                        or record.get("caption")
                        or record.get("title")
                        or ""
                    )
                    if text:
                        documents.append(
                            KnowledgeDocument(
                                doc_id=normalize_text(record.get("id") or record.get("doc_id") or record.get("kid") or f"{path.stem}:{idx}"),
                                text=text,
                                source=normalize_text(record.get("source") or str(path)),
                                metadata={"raw": record, "path": str(path), "title": record.get("title"), "timestamp": record.get("rev_timestamp")},
                            )
                        )
                    if max_documents is not None and len(documents) >= max_documents:
                        return documents[:max_documents]
            elif path.is_file():
                documents.append(KnowledgeDocument(doc_id=path.stem, text=path.read_text(encoding="utf-8", errors="ignore"), source=str(path)))
            if max_documents is not None and len(documents) >= max_documents:
                return documents[:max_documents]
        return documents

    def _expand_paths(self, corpus_paths: list[str | Path]) -> list[Path]:
        paths: list[Path] = []
        for raw_path in corpus_paths:
            path = Path(raw_path)
            if any(char in str(raw_path) for char in ["*", "?", "["]):
                paths.extend(sorted(Path(".").glob(str(raw_path))))
            elif path.is_dir():
                paths.extend(sorted(path.rglob("*.jsonl")))
                paths.extend(sorted(path.rglob("*.txt")))
            else:
                paths.append(path)
        return paths

    def _with_score(self, document: KnowledgeDocument, score: float, score_key: str) -> KnowledgeDocument:
        metadata = dict(document.metadata)
        metadata[score_key] = float(score)
        metadata["retrieval_score"] = float(score)
        return KnowledgeDocument(document.doc_id, document.text, document.source, metadata)

    def _fallback_results(self, query: str, top_k: int) -> list[KnowledgeDocument]:
        keywords = keyword_candidates(query, limit=5)
        topic = ", ".join(keywords) if keywords else normalize_text(query)[:80] or "the meme"
        templates = [
            f"Background cue: the meme text mentions {topic}, which may indicate a target, event, or cultural reference.",
            f"Interpretive hypothesis: compare the visual setup with the phrase '{normalize_text(query)[:120]}' to identify sarcasm or contrast.",
            f"Safety lens: check whether the joke assigns blame, inferiority, threat, or ridicule to a person or group connected to {topic}.",
        ]
        return [
            KnowledgeDocument(
                doc_id=f"fallback:{idx}",
                text=text,
                source="fallback",
                metadata={"retrieval_score": max(0.1, 0.5 - idx * 0.1), "fallback": True},
            )
            for idx, text in enumerate(templates[:top_k])
        ]
