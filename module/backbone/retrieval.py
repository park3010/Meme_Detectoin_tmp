"""Retrieval and pairwise scoring adapters."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from utils.io import read_jsonl
from utils.retrieval_utils import bm25_like_score, lexical_retrieval_score, reciprocal_rank_fusion
from utils.tensor_utils import hashed_vector
from utils.text_utils import jaccard_similarity, keyword_candidates, normalize_text


# =============================================================================
# Local retriever adapter
# =============================================================================

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
        index_root: str | Path | None = None,
        query_cache_root: str | Path | None = None,
    ) -> None:
        self.dense_dim = dense_dim
        self.documents = self._load_corpus(corpus_paths or [], max_documents=max_documents)
        self.avg_doc_len = sum(len(doc.text.split()) for doc in self.documents) / max(1, len(self.documents))
        self.index_root = Path(index_root) if index_root else None
        indexed_embeddings = self._load_dense_index()
        if self.index_root is not None and self.documents and indexed_embeddings is None:
            raise RuntimeError(f"Verified dense retrieval index could not be loaded from {self.index_root}")
        self.document_embeddings = indexed_embeddings or [
            hashed_vector(document.text, dim=dense_dim) for document in self.documents
        ]
        self.sparse_index_loaded = self._verify_sparse_index()
        if self.index_root is not None and self.documents and not self.sparse_index_loaded:
            raise RuntimeError(f"Verified sparse retrieval index could not be loaded from {self.index_root}")
        self.fallback_candidates = fallback_candidates
        self.query_cache_root = Path(query_cache_root) if query_cache_root else None

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

    def search(
        self,
        query: str,
        top_k: int = 8,
        *,
        query_context: dict[str, str] | None = None,
    ) -> list[KnowledgeDocument]:
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
        if not results and self.fallback_candidates:
            results = self._fallback_results(query, top_k)
        self._write_query_cache(query, results, query_context=query_context)
        return results

    def set_query_cache(self, root: str | Path | None) -> None:
        """Set a run-scoped query-result cache; canonical indexes stay read-only."""

        self.query_cache_root = Path(root) if root else None

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
                                doc_id=normalize_text(record.get("document_id") or record.get("id") or record.get("doc_id") or record.get("kid") or f"{path.stem}:{idx}"),
                                text=text,
                                source=normalize_text(record.get("source_name") or record.get("source") or str(path)),
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

    def _load_dense_index(self) -> list[torch.Tensor] | None:
        if self.index_root is None or not self.documents:
            return None
        dense_root = self.index_root / "dense"
        ids_path = dense_root / "document_ids.jsonl"
        vectors_path = dense_root / "embeddings.f32"
        metadata_path = dense_root / "metadata.json"
        if not ids_path.is_file() or not vectors_path.is_file() or not metadata_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            ids = [str(row.get("document_id", "")) for row in read_jsonl(ids_path)]
            expected_ids = [document.doc_id for document in self.documents]
            if ids[: len(expected_ids)] != expected_ids or int(metadata.get("dimension", -1)) != self.dense_dim:
                return None
            data = vectors_path.read_bytes()
            expected_bytes = len(ids) * self.dense_dim * 4
            if len(data) != expected_bytes:
                return None
            values = struct.unpack(f"<{len(ids) * self.dense_dim}f", data)
            matrix = torch.tensor(values, dtype=torch.float32).reshape(len(ids), self.dense_dim)
            return [row for row in matrix[: len(expected_ids)]]
        except (OSError, ValueError, json.JSONDecodeError, struct.error):
            return None

    def _verify_sparse_index(self) -> bool:
        if self.index_root is None:
            return False
        sparse_root = self.index_root / "sparse"
        metadata_path = sparse_root / "metadata.json"
        terms_path = sparse_root / "document_terms.jsonl"
        if not metadata_path.is_file() or not terms_path.is_file():
            return False
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            return int(metadata.get("document_count", -1)) >= len(self.documents)
        except (OSError, ValueError, json.JSONDecodeError):
            return False

    def _write_query_cache(
        self,
        query: str,
        results: list[KnowledgeDocument],
        *,
        query_context: dict[str, str] | None,
    ) -> None:
        if self.query_cache_root is None or not query_context:
            return
        dataset_name = normalize_text(query_context.get("dataset_name", "unknown")) or "unknown"
        role = "fhm" if dataset_name in {"facebook", "fhm"} else dataset_name
        cache_path = self.query_cache_root / role / "queries.jsonl"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "dataset_name": dataset_name,
            "sample_id": normalize_text(query_context.get("sample_id", "")),
            "query": query,
            "results": [
                {
                    "document_id": document.doc_id,
                    "source": document.source,
                    "retrieval_score": float(document.metadata.get("retrieval_score", 0.0)),
                }
                for document in results
            ],
            "contains_labels": False,
        }
        with cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

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


# =============================================================================
# Cross-encoder adapter
# =============================================================================

class CrossEncoderAdapter:
    """Simple pair scorer standing in for a trained cross encoder."""

    def score(self, query: str, candidate: str) -> float:
        """Return a pairwise semantic proxy score in [0, 1]."""

        score = jaccard_similarity(query, candidate)
        keywords = keyword_candidates(query, limit=8)
        if keywords:
            coverage = sum(1 for key in keywords if key in candidate.lower()) / len(keywords)
            score = 0.55 * score + 0.45 * coverage
        return max(0.0, min(1.0, score))

    def classify_support(self, claim: str, evidence: str) -> tuple[str, float]:
        """Classify evidence as support, contradiction, or insufficient."""

        score = self.score(claim, evidence)
        negative_terms = {"not", "never", "false", "fake", "hoax", "against", "contradict"}
        evidence_terms = set(evidence.lower().split())
        if score >= 0.28:
            label = "contradict" if negative_terms & evidence_terms else "support"
        elif score >= 0.12:
            label = "insufficient"
        else:
            label = "irrelevant"
        return label, score


__all__ = ["KnowledgeDocument", "LocalRetrieverAdapter", "CrossEncoderAdapter"]
