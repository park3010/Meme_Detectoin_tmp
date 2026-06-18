"""Lightweight retrieval scoring helpers."""

from __future__ import annotations

from collections import defaultdict
from math import log

from utils.text_utils import jaccard_similarity, keyword_candidates, tokenize


def lexical_retrieval_score(query: str, document: str) -> float:
    """Combine Jaccard overlap with keyword coverage."""

    jaccard = jaccard_similarity(query, document)
    keywords = keyword_candidates(query, limit=10)
    if not keywords:
        return jaccard
    doc_lower = document.lower()
    coverage = sum(1 for key in keywords if key.lower() in doc_lower) / len(keywords)
    return 0.65 * jaccard + 0.35 * coverage


def bm25_like_score(query: str, document: str, avg_doc_len: float = 80.0) -> float:
    """Compute a lightweight BM25-style score without external dependencies."""

    query_terms = keyword_candidates(query, limit=16)
    doc_terms = tokenize(document)
    if not query_terms or not doc_terms:
        return 0.0
    freqs: dict[str, int] = defaultdict(int)
    for term in doc_terms:
        freqs[term] += 1
    k1 = 1.2
    b = 0.75
    doc_len = len(doc_terms)
    score = 0.0
    for term in query_terms:
        tf = freqs.get(term.lower(), 0)
        if tf == 0:
            continue
        idf_proxy = log(1.0 + 1.0 / (1.0 + query_terms.count(term)))
        denom = tf + k1 * (1.0 - b + b * doc_len / max(avg_doc_len, 1.0))
        score += idf_proxy * (tf * (k1 + 1.0)) / max(denom, 1e-6)
    return float(score)


def reciprocal_rank_fusion(rankings: list[list[tuple[str, float]]], k: int = 60) -> dict[str, float]:
    """Fuse ranked document ids using reciprocal rank fusion."""

    fused: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            fused[doc_id] += 1.0 / (k + rank)
    return dict(fused)
