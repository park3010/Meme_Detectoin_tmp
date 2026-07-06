"""Evidence attribution and error-case selection utilities."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from experiments.evaluation import compute_harmfulness_metrics, evaluate_structured_predictions
from experiments.prediction_io import load_prediction_records
from experiments.progress import ProgressConfig, progress_iter
from utils.io import write_json, write_jsonl


CASE_TARGETS = {"true_positive": 3, "true_negative": 2, "false_positive": 3, "false_negative": 3}
SUBSETS = [
    "image_text_incongruity",
    "implicit_target",
    "external_knowledge_needed",
    "sarcasm_irony",
    "metaphor_symbolic",
    "ocr_heavy",
    "visual_symbol_heavy",
]


def select_error_cases(
    dataset: str = "all",
    model: str = "ours_full",
    seed: int = 42,
    result_root: str = "result",
    output_root: str | None = None,
    disable_tqdm: bool = False,
    progress: ProgressConfig | None = None,
) -> list[dict[str, Any]]:
    """Select representative TP/TN/FP/FN cases from saved predictions."""

    records = _load_records(dataset, model, seed, result_root)
    selected: list[dict[str, Any]] = []
    progress_config = progress or ProgressConfig(disable=True if disable_tqdm else None)
    for category, count in CASE_TARGETS.items():
        cases = [
            case_record(record, category)
            for record in progress_iter(
                records,
                desc=f"select {category}",
                config=progress_config,
                position=2,
                leave=progress_config.leave_batch,
            )
            if _case_category(record) == category
        ]
        selected.extend(cases[:count])
    out_root = Path(output_root or result_root) / "analysis" / "error_cases"
    write_jsonl(out_root / "error_cases.jsonl", selected, compact_tensors=True)
    for case in selected:
        folder = out_root / str(case["case_type"])
        write_json(folder / f"{case['sample_id']}.json", case)
    return selected


def export_case_visualization_data(
    dataset: str = "all",
    model: str = "ours_full",
    seed: int = 42,
    result_root: str = "result",
    disable_tqdm: bool = False,
    progress: ProgressConfig | None = None,
) -> Path:
    """Export all selected case data needed by downstream visualization scripts."""

    cases = select_error_cases(
        dataset=dataset,
        model=model,
        seed=seed,
        result_root=result_root,
        disable_tqdm=disable_tqdm,
        progress=progress,
    )
    path = Path(result_root) / "analysis" / "error_cases" / "case_visualization_data.jsonl"
    write_jsonl(path, cases, compact_tensors=True)
    return path


def case_record(record: dict[str, Any], category: str | None = None) -> dict[str, Any]:
    """Convert one prediction row into a richer case-study record."""

    category = category or _case_category(record)
    stage_metadata = record.get("stage_metadata", {}) or {}
    supporting = record.get("supporting_evidence", {}) or {}
    return {
        "sample_id": record.get("sample_id"),
        "dataset_name": record.get("dataset_name"),
        "case_type": category,
        "image_path": record.get("image_path"),
        "ocr_text": record.get("ocr_text_full", ""),
        "gold_harmfulness": record.get("gold_harmfulness") or record.get("gold_label"),
        "predicted_harmfulness": record.get("pred_harmfulness") or record.get("pred_label"),
        "gold_target": record.get("gold_target"),
        "predicted_target": record.get("target"),
        "gold_intent": record.get("gold_intent"),
        "predicted_intent": record.get("intent"),
        "gold_tactic": record.get("gold_tactic"),
        "predicted_tactic": record.get("tactic"),
        "selected_internal_evidence": supporting.get("internal", []),
        "selected_external_evidence": supporting.get("external", []),
        "verifier_accepted_knowledge": supporting.get("external", []),
        "verifier_rejected_knowledge": record.get("rejected_knowledge", []),
        "gate_values": _gate_values(stage_metadata),
        "rationale": record.get("rationale", ""),
        "failure_reason_category": failure_reason(record),
    }


def failure_reason(record: dict[str, Any]) -> str:
    """Heuristically label likely failure reason for a prediction record."""

    if record.get("gold_label") == record.get("pred_label"):
        return "not_failure"
    ocr = str(record.get("ocr_text_full", ""))
    supporting = record.get("supporting_evidence", {}) or {}
    stage_metadata = record.get("stage_metadata", {}) or {}
    stage_b = stage_metadata.get("stage_b", {}) or {}
    stage_c = stage_metadata.get("stage_c", {}) or {}
    if not ocr.strip():
        return "OCR failure"
    if not supporting.get("internal"):
        return "symbol detection failure"
    if int(stage_b.get("retrieved_count", 0) or 0) == 0:
        return "retrieval failure"
    if int(stage_c.get("filtered_candidate_count", 0) or 0) == 0 and int(stage_b.get("retrieved_count", 0) or 0) > 0:
        return "verifier false rejection"
    if supporting.get("external") and record.get("gold_label") == 0 and record.get("pred_label") == 1:
        return "verifier false acceptance"
    if record.get("gold_target") and record.get("target") and record.get("gold_target") != record.get("target"):
        return "reasoning/fusion error"
    return "unknown"


def _load_records(dataset: str, model: str, seed: int, result_root: str) -> list[dict[str, Any]]:
    root = Path(result_root) / "predictions"
    paths = sorted(root.glob(f"*/{model}/{seed}/final_predictions.jsonl")) if dataset == "all" else [root / dataset / model / str(seed) / "final_predictions.jsonl"]
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(load_prediction_records(path))
    return records


def _case_category(record: dict[str, Any]) -> str:
    gold = record.get("gold_label")
    pred = record.get("pred_label")
    if gold == 1 and pred == 1:
        return "true_positive"
    if gold == 0 and pred == 0:
        return "true_negative"
    if gold == 0 and pred == 1:
        return "false_positive"
    if gold == 1 and pred == 0:
        return "false_negative"
    return "unknown"


def _gate_values(stage_metadata: dict[str, Any]) -> dict[str, Any]:
    stage_d = stage_metadata.get("stage_d", {}) if isinstance(stage_metadata, dict) else {}
    return {
        "knowledge_need": stage_d.get("knowledge_need"),
        "gate_mode": stage_d.get("gate_mode"),
        "task_support_used": stage_d.get("task_support_used"),
    }


# =============================================================================
# Subset slicing and summary
# =============================================================================

def run_subset_analysis(
    dataset: str = "all",
    model: str = "ours_full",
    seed: int = 42,
    result_root: str = "result",
) -> list[dict[str, Any]]:
    """Analyze saved predictions on difficult subsets."""

    records = _load_records(dataset, model, seed, result_root)
    rows = subset_metric_rows(records, dataset, model, seed)
    path = Path(result_root) / "metrics" / "subset_analysis.csv"
    _append_rows(path, rows)
    return rows


def subset_metric_rows(records: list[dict[str, Any]], dataset: str, model: str, seed: int) -> list[dict[str, Any]]:
    """Build per-subset metric rows from prediction records."""

    rows: list[dict[str, Any]] = []
    for subset in SUBSETS:
        subset_records = [record for record in records if subset in assign_subsets(record)]
        metrics = _metrics(subset_records)
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "seed": seed,
                "subset": subset,
                "sample_count": len(subset_records),
                "harmfulness_macro_f1": metrics.get("macro_f1") or metrics.get("harmfulness_macro_f1"),
                "target_macro_f1": metrics.get("target_granularity_macro_f1"),
                "intent_macro_f1": metrics.get("intent_primary_macro_f1"),
                "tactic_macro_f1": metrics.get("tactic_multimodal_relation_macro_f1"),
                "error_rate": metrics.get("error_rate"),
            }
        )
    return rows


def assign_subsets(record: dict[str, Any]) -> list[str]:
    """Assign one prediction record to difficult subsets using annotations and fallbacks."""

    subsets: list[str] = []
    tactic = record.get("gold_tactic", {}) or {}
    target = record.get("gold_target", {}) or {}
    intent = record.get("gold_intent", {}) or {}
    predicted_tactic = record.get("tactic", {}) or {}
    stage_a = (record.get("stage_metadata", {}) or {}).get("stage_a", {}) or {}
    stage_d = (record.get("stage_metadata", {}) or {}).get("stage_d", {}) or {}
    relation = str(tactic.get("tactic_multimodal_relation") or predicted_tactic.get("multimodal_relation") or "").lower()
    rhetorical = [str(item).lower() for item in _as_list(tactic.get("tactic_rhetorical")) + _as_list(predicted_tactic.get("rhetorical"))]
    ocr_text = str(record.get("ocr_text_full", ""))
    evidence_text = " ".join(str(item) for item in _as_list(record.get("gold_evidence_text")))

    if any(term in relation for term in ["incongruent", "contrast", "contradict"]):
        subsets.append("image_text_incongruity")
    if str(target.get("target_presence", "")).lower() == "implicit":
        subsets.append("implicit_target")
    if bool(intent.get("background_knowledge_needed")) or float(stage_d.get("knowledge_need", 0.0) or 0.0) >= 0.45:
        subsets.append("external_knowledge_needed")
    if any("sarcasm" in item or "irony" in item for item in rhetorical) or any(cue in ocr_text.lower() for cue in ["yeah right", "sure", "totally"]):
        subsets.append("sarcasm_irony")
    if any("metaphor" in item or "symbol" in item for item in rhetorical) or "symbol" in evidence_text.lower():
        subsets.append("metaphor_symbolic")
    if len(ocr_text.split()) >= 35:
        subsets.append("ocr_heavy")
    if int(stage_a.get("roi_count", 0) or 0) >= 2 or any("symbol" in str(item).lower() for group in (record.get("supporting_evidence", {}) or {}).values() for item in group):
        subsets.append("visual_symbol_heavy")
    return subsets


def _metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"error_rate": None}
    labeled = [record for record in records if record.get("gold_label") is not None]
    harmfulness = compute_harmfulness_metrics(
        [record["gold_label"] for record in labeled],
        [record["pred_label"] for record in labeled],
        [record.get("prob_harmful", 0.0) for record in labeled],
    )
    structured = evaluate_structured_predictions(records)
    errors = sum(1 for record in labeled if record.get("gold_label") != record.get("pred_label"))
    return {**harmfulness, **structured, "error_rate": errors / max(1, len(labeled))}


def _load_records(dataset: str, model: str, seed: int, result_root: str) -> list[dict[str, Any]]:
    root = Path(result_root) / "predictions"
    paths = sorted(root.glob(f"*/{model}/{seed}/final_predictions.jsonl")) if dataset == "all" else [root / dataset / model / str(seed) / "final_predictions.jsonl"]
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(load_prediction_records(path))
    return records


def _append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    columns = list(rows[0]) if rows else ["dataset", "model", "seed", "subset"]
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _as_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    return value if isinstance(value, list) else [value]
