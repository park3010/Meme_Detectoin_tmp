"""Export auditable held-out FHM error-analysis packages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.io import read_jsonl, write_json, write_jsonl


ERROR_TAXONOMY = [
    "Stage A error",
    "Stage B error",
    "Stage C error",
    "Stage D error",
    "Stage E error",
    "multiple-stage error",
    "annotation ambiguity",
    "cultural-context error",
    "uncertain",
]


def export_research_error_cases(
    *,
    output_root: str = "result",
    suite: str = "harmeme_to_fhm_1seed",
    experiment_id: str = "ours_full",
    seed: int = 42,
    per_category: int = 10,
) -> dict[str, Any]:
    """Select TP/TN/FP/FN records and preserve analysis fields verbatim."""

    run_dir = Path(output_root) / "research_runs" / suite / experiment_id / f"seed_{seed}"
    records = read_jsonl(run_dir / "test_predictions.jsonl")
    categories = (
        "true_positive",
        "true_negative",
        "false_positive",
        "false_negative",
        "target_error",
        "intent_error",
        "tactic_error",
        "multimodal_relation_error",
        "high_confidence_error",
        "cross_modal_confounder",
        "retrieval_heavy",
        "verifier_rejection",
    )
    buckets: dict[str, list[dict[str, Any]]] = {key: [] for key in categories}
    for record in records:
        gold = record.get("gold_label")
        pred = record.get("pred_label")
        if gold not in {0, 1} or pred not in {0, 1}:
            continue
        category = (
            "true_positive" if gold == pred == 1 else
            "true_negative" if gold == pred == 0 else
            "false_positive" if gold == 0 and pred == 1 else
            "false_negative"
        )
        exported = dict(record)
        exported["image_path"] = _safe_relative_path(record.get("image_path"))
        exported["case_category"] = category
        exported["failure_reason_category"] = "uncertain" if category.startswith("false") else None
        exported["causal_stage_attribution_source"] = "human_review_required"
        exported["failure_reason_taxonomy"] = ERROR_TAXONOMY
        _append(buckets, category, exported, per_category)
        gold_target = record.get("gold_target", {}) or {}
        pred_target = record.get("target", {}) or {}
        if _different(gold_target.get("target_presence"), pred_target.get("presence")) or _different(gold_target.get("target_granularity"), pred_target.get("granularity")):
            _append(buckets, "target_error", exported, per_category)
        if _different((record.get("gold_intent", {}) or {}).get("intent_primary"), (record.get("intent", {}) or {}).get("primary")):
            _append(buckets, "intent_error", exported, per_category)
        if _set_different((record.get("gold_tactic", {}) or {}).get("tactic_rhetorical"), (record.get("tactic", {}) or {}).get("rhetorical")):
            _append(buckets, "tactic_error", exported, per_category)
        if _different((record.get("gold_tactic", {}) or {}).get("tactic_multimodal_relation"), (record.get("tactic", {}) or {}).get("multimodal_relation")):
            _append(buckets, "multimodal_relation_error", exported, per_category)
        probability = _float(record.get("prob_harmful"))
        if gold != pred and probability is not None and (probability >= 0.8 or probability <= 0.2):
            _append(buckets, "high_confidence_error", exported, per_category)
        stage_a = (record.get("stage_metadata", {}) or {}).get("stage_a", {}) or {}
        relation = str(stage_a.get("auxiliary_labels", {}).get("multimodal_relation", "")) if isinstance(stage_a, dict) else ""
        if gold != pred and relation in {"incongruent", "cross_modal_implication"}:
            _append(buckets, "cross_modal_confounder", exported, per_category)
        external = (record.get("supporting_evidence", {}) or {}).get("external", []) or []
        if len(external) >= 3:
            _append(buckets, "retrieval_heavy", exported, per_category)
        stage_c = (record.get("stage_metadata", {}) or {}).get("stage_c", {}) or {}
        if isinstance(stage_c, dict) and int(stage_c.get("rejected_count", 0) or 0) > 0:
            _append(buckets, "verifier_rejection", exported, per_category)
    root = Path(output_root) / "analysis" / "research_error_cases" / suite / experiment_id / f"seed_{seed}"
    all_rows = []
    for category, rows in buckets.items():
        write_jsonl(root / f"{category}.jsonl", rows)
        for row in rows:
            sample_id = str(row.get("sample_id", len(all_rows)))
            write_json(root / category / f"{sample_id}.json", row)
        all_rows.extend(rows)
    write_jsonl(root / "error_cases.jsonl", all_rows)
    manifest = {
        "schema_version": "research_error_case_export_v1",
        "source_run": str(run_dir),
        "counts": {key: len(value) for key, value in buckets.items()},
        "failure_taxonomy": ERROR_TAXONOMY,
        "human_failure_labels_present": False,
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def _append(buckets: dict[str, list[dict[str, Any]]], category: str, row: dict[str, Any], limit: int) -> None:
    if len(buckets[category]) < limit and not any(item.get("sample_id") == row.get("sample_id") for item in buckets[category]):
        buckets[category].append(dict(row, analysis_category=category))


def _different(gold: Any, pred: Any) -> bool:
    return gold not in {None, "", "unknown", "ambiguous"} and str(gold) != str(pred)


def _set_different(gold: Any, pred: Any) -> bool:
    if gold is None or gold == "":
        return False
    gold_values = set(gold if isinstance(gold, list) else [gold]) - {"none", "unknown", "ambiguous"}
    pred_values = set(pred if isinstance(pred, list) else [pred]) - {"none"}
    return bool(gold_values) and gold_values != pred_values


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_relative_path(value: Any) -> str | None:
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return path.name


__all__ = ["ERROR_TAXONOMY", "export_research_error_cases"]
