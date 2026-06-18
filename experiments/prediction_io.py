"""Prediction serialization helpers for full-framework experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from experiments.splits import label_to_int
from module.losses import extract_supervision_from_annotation
from utils.io import read_jsonl, write_json, write_jsonl
from utils.tensor_utils import tensor_to_python


def stage_outputs_to_prediction_record(
    sample: dict[str, Any],
    outputs: dict[str, Any],
    model_name: str,
    seed: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert pipeline stage outputs into a compact experiment prediction row."""

    stage_e = outputs["stage_e"]
    structured = stage_e.structured_prediction
    supervision = extract_supervision_from_annotation(sample)
    gold_label = label_to_int(supervision.get("harmfulness"))
    if gold_label is None:
        gold_label = label_to_int(sample.get("raw_label"))
    harm_scores = structured.get("harmfulness", {}).get("scores", {})
    prob_harmful = float(harm_scores.get("harmful", structured.get("harmfulness", {}).get("score", 0.0)))
    pred_label = 1 if structured.get("harmfulness", {}).get("label") == "harmful" else 0
    normalized_annotation = sample.get("normalized_annotation") or {}
    source_annotation = normalized_annotation.get("source_annotation", {}) if isinstance(normalized_annotation, dict) else {}

    record = {
        "sample_id": sample.get("sample_id"),
        "dataset_name": sample.get("dataset_name"),
        "model_name": model_name,
        "seed": seed,
        "image_path": sample.get("image_path"),
        "ocr_text_full": sample.get("ocr_text_full") or sample.get("ocr_text") or "",
        "gold_label": gold_label,
        "pred_label": pred_label,
        "prob_harmful": prob_harmful,
        "gold_harmfulness": supervision.get("harmfulness"),
        "pred_harmfulness": structured.get("harmfulness", {}).get("label"),
        "harmfulness_score": structured.get("harmfulness", {}).get("score"),
        "target": structured.get("target", {}),
        "gold_target": {
            "target_presence": supervision.get("target_presence"),
            "target_granularity": supervision.get("target_granularity"),
            "protected_attribute": supervision.get("target_attributes"),
        },
        "intent": structured.get("intent", {}),
        "gold_intent": {
            "intent_primary": supervision.get("intent_primary"),
            "stance": supervision.get("stance"),
            "secondary_intent": supervision.get("secondary_intent"),
            "background_knowledge_needed": supervision.get("background_knowledge_needed"),
        },
        "tactic": structured.get("tactic", {}),
        "gold_tactic": {
            "tactic_rhetorical": supervision.get("tactic_rhetorical"),
            "tactic_multimodal_relation": supervision.get("tactic_multimodal_relation"),
        },
        "gold_evidence_text": _evidence_text_list(supervision.get("evidence_text", [])),
        "gold_label_strings": sample.get("label_strings", {}),
        "gold_audit_flags": sample.get("audit_flags", []),
        "gold_sample_weight": sample.get("sample_weight", supervision.get("sample_weight", 1.0)),
        "gold_normalized_schema_version": source_annotation.get("annotation_schema_version"),
        "supporting_evidence": structured.get("supporting_evidence", {}),
        "rationale": structured.get("rationale", ""),
        "stage_metadata": compact_stage_metadata(outputs),
    }
    if extra:
        record.update(extra)
    return tensor_to_python(record, max_elements=12)


def compact_stage_metadata(outputs: dict[str, Any]) -> dict[str, Any]:
    """Collect compact metadata useful for analysis without large tensors."""

    metadata: dict[str, Any] = {}
    for stage_name in ["stage_a", "stage_b", "stage_c", "stage_d", "stage_e"]:
        output = outputs.get(stage_name)
        if output is None or not hasattr(output, "metadata"):
            continue
        metadata[stage_name] = tensor_to_python(output.metadata, max_elements=8)
    return metadata


def save_predictions_and_metrics(
    output_dir: str | Path,
    predictions: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    """Save final_predictions.jsonl and metrics.json under output_dir."""

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    write_jsonl(path / "final_predictions.jsonl", predictions)
    write_json(path / "metrics.json", metrics)


def load_prediction_records(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL prediction records."""

    return read_jsonl(path)


def detach_loss_value(value: torch.Tensor | float | int | None) -> float:
    """Convert an optional tensor scalar into a float."""

    if value is None:
        return 0.0
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def _evidence_text_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, dict):
        return [str(item) for item in value.values() if str(item).strip()]
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]
