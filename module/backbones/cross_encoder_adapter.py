"""Pairwise relevance/support verifier adapter."""

from __future__ import annotations

from utils.text_utils import jaccard_similarity, keyword_candidates


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
