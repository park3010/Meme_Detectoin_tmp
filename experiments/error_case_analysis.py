"""Evidence attribution and error-case selection utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.prediction_io import load_prediction_records
from experiments.progress import progress_iter
from utils.io import write_json, write_jsonl


CASE_TARGETS = {"true_positive": 3, "true_negative": 2, "false_positive": 3, "false_negative": 3}


def select_error_cases(
    dataset: str = "all",
    model: str = "ours_full",
    seed: int = 42,
    result_root: str = "result",
    output_root: str | None = None,
    disable_tqdm: bool = False,
) -> list[dict[str, Any]]:
    """Select representative TP/TN/FP/FN cases from saved predictions."""

    records = _load_records(dataset, model, seed, result_root)
    selected: list[dict[str, Any]] = []
    for category, count in CASE_TARGETS.items():
        cases = [
            case_record(record, category)
            for record in progress_iter(records, desc=f"select {category}", leave=False, disable=disable_tqdm)
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
) -> Path:
    """Export all selected case data needed by downstream visualization scripts."""

    cases = select_error_cases(
        dataset=dataset,
        model=model,
        seed=seed,
        result_root=result_root,
        disable_tqdm=disable_tqdm,
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
