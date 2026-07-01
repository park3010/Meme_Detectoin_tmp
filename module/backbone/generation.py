"""Generation adapters."""

from __future__ import annotations

from utils.text_utils import keyword_candidates, normalize_text


# =============================================================================
# Template generator adapter
# =============================================================================

class GeneratorAdapter:
    """Template generator used until a constrained generation model is added."""

    def generate_hypotheses(self, text: str, evidence_summary: str = "", max_items: int = 3) -> list[str]:
        """Generate deterministic interpretation hypotheses."""

        clean = normalize_text(text)
        keywords = keyword_candidates(f"{clean} {evidence_summary}", limit=5)
        topic = ", ".join(keywords) if keywords else "the meme"
        hypotheses = [
            f"The meme may rely on a shared reference involving {topic}.",
            "The intended effect may come from contrast between OCR text and the image context.",
            "The harmfulness judgment should check whether the target is ridiculed, blamed, or dehumanized.",
        ]
        if evidence_summary:
            hypotheses.append(f"Internal evidence summary to verify: {normalize_text(evidence_summary)[:160]}.")
        return hypotheses[:max_items]


__all__ = ["GeneratorAdapter"]
