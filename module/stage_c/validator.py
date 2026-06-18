"""Credibility, temporal, and cultural validation for Stage C."""

from __future__ import annotations

from datetime import datetime, timezone

from module.stage_b.schemas import KnowledgeCandidate
from utils.text_utils import language_hint


class CredibilityTemporalValidator:
    """Apply simple source and compatibility checks."""

    def validate(self, candidate: KnowledgeCandidate, internal_text: str = "") -> tuple[float, dict[str, float]]:
        """Return validity score and component scores."""

        source = candidate.source.lower()
        path = str(candidate.metadata.get("path", "")).lower()
        if source == "fallback":
            credibility = 0.45
        elif "wikipedia" in source or "wiki" in source or "wiki" in path:
            credibility = 0.78
        elif "rag" in source or "rag" in path:
            credibility = 0.72
        elif source == "template_generator":
            credibility = 0.5
        else:
            credibility = 0.62
        temporal = _temporal_score(candidate)
        internal_lang = language_hint(internal_text)
        candidate_lang = language_hint(candidate.text)
        language_match = 1.0 if internal_lang == "unknown" or candidate_lang == "unknown" or internal_lang == candidate_lang else 0.55
        cultural = 0.68 if any(word in candidate.text.lower() for word in ["meme", "culture", "context", "reference"]) else 0.55
        cultural = 0.5 * cultural + 0.5 * language_match
        validity = 0.5 * credibility + 0.25 * temporal + 0.25 * cultural
        return validity, {"credibility": credibility, "temporal": temporal, "cultural": cultural, "language_match": language_match}


def _temporal_score(candidate: KnowledgeCandidate) -> float:
    timestamp = candidate.metadata.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp:
        raw = candidate.metadata.get("raw", {})
        timestamp = raw.get("rev_timestamp") if isinstance(raw, dict) else None
    if not isinstance(timestamp, str) or not timestamp:
        return 0.64
    try:
        value = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return 0.6
    now = datetime.now(timezone.utc)
    age_days = max(0, (now - value).days)
    if age_days < 365:
        return 0.8
    if age_days < 365 * 5:
        return 0.68
    return 0.55
