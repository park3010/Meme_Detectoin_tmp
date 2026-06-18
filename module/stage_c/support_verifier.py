"""Support/contradiction verification for Stage C."""

from __future__ import annotations

from module.backbones.cross_encoder_adapter import CrossEncoderAdapter


class SupportContradictionVerifier:
    """Classify whether a knowledge item supports the internal hypothesis."""

    def __init__(self) -> None:
        self.cross_encoder = CrossEncoderAdapter()

    def verify(self, internal_summary: str, candidate_text: str) -> tuple[str, float]:
        """Return support label and confidence-like score."""

        label, score = self.cross_encoder.classify_support(internal_summary, candidate_text)
        if label == "irrelevant":
            label = "insufficient"
        return label, score

    def verify_claims(self, claims: dict[str, str], candidate_text: str) -> dict[str, dict[str, float | str]]:
        """Verify target/intent/tactic claims independently."""

        results: dict[str, dict[str, float | str]] = {}
        for claim_type, claim in claims.items():
            if not claim.strip():
                results[claim_type] = {"label": "insufficient", "score": 0.0}
                continue
            label, score = self.verify(claim, candidate_text)
            results[claim_type] = {"label": label, "score": float(score)}
        return results
