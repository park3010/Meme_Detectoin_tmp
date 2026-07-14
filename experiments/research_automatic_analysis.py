"""Automatic, explicitly proxy-only analysis of evidence and rationales."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from utils.io import read_jsonl, write_json
from utils.text_utils import jaccard_similarity


def analyze_prediction_artifacts(prediction_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Collect supported automatic fields without claiming human faithfulness."""

    records = read_jsonl(prediction_path)
    internal_counts = []
    external_counts = []
    rationale_lengths = []
    overlaps = []
    provenance_complete = []
    origins: Counter[str] = Counter()
    for record in records:
        supporting = record.get("supporting_evidence", {}) or {}
        internal = list(supporting.get("internal", []) or [])
        external = list(supporting.get("external", []) or [])
        internal_counts.append(len(internal))
        external_counts.append(len(external))
        rationale = str(record.get("rationale", ""))
        rationale_lengths.append(len(rationale.split()))
        evidence_text = " ".join(_item_text(item) for item in [*internal, *external]).strip()
        overlaps.append(jaccard_similarity(rationale, evidence_text) if rationale and evidence_text else 0.0)
        items = [item for item in [*internal, *external] if isinstance(item, dict)]
        provenance_complete.append(
            all(bool(item.get("evidence_id") or item.get("candidate_id")) and bool(item.get("text")) for item in items)
            if items
            else False
        )
        for item in external:
            if isinstance(item, dict):
                origins[str(item.get("candidate_origin") or "unknown")] += 1
    result = {
        "schema_version": "automatic_evidence_rationale_proxy_v1",
        "analysis_provenance": "automatic_proxy_not_human_faithfulness",
        "sample_count": len(records),
        "selected_internal_evidence_count_mean": _mean(internal_counts),
        "selected_external_evidence_count_mean": _mean(external_counts),
        "internal_external_evidence_ratio": sum(internal_counts) / sum(external_counts) if sum(external_counts) else None,
        "evidence_rationale_lexical_overlap_mean": _mean(overlaps),
        "rationale_length_tokens_mean": _mean(rationale_lengths),
        "provenance_completeness_ratio": sum(provenance_complete) / len(provenance_complete) if provenance_complete else None,
        "candidate_origin_distribution": dict(sorted(origins.items())),
        "retrieved_candidate_count": _origin_count(origins, "retrieved"),
        "generated_candidate_count": _origin_count(origins, "generated"),
        "fallback_candidate_count": _origin_count(origins, "fallback"),
        "verified_knowledge_count": sum(external_counts) if sum(external_counts) else None,
        "rejection_count": None,
        "unsupported_claim_heuristic": None,
        "human_faithfulness_score": None,
    }
    write_json(output_path, result)
    return result


def _origin_count(origins: Counter[str], needle: str) -> int | None:
    matches = sum(count for name, count in origins.items() if needle in name.lower())
    return matches if origins else None


def _item_text(item: Any) -> str:
    return str(item.get("text", "")) if isinstance(item, dict) else str(item)


def _mean(values: list[float | int]) -> float | None:
    return mean(values) if values else None


__all__ = ["analyze_prediction_artifacts"]
