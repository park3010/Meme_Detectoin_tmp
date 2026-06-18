"""Build normalized silver labels from LLM-generated annotation JSONL files."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from dataset import MemeDataset
from experiments.progress import progress_iter
from utils.annotation_utils import (
    confidence_score,
    get_block,
    is_missing,
    map_raw_harmfulness,
    normalize_label,
    normalize_label_list,
    parse_bool,
    raw_annotation_keys,
    sample_to_dict,
    unwrap_annotation,
)
from utils.io import load_yaml, write_jsonl


DEFAULT_DATASETS = ["harm_c", "harm_p", "facebook", "memotion"]


def load_normalization_config(path: str | Path = "configs/annotation_normalization.yaml") -> dict[str, Any]:
    """Load annotation normalization YAML config."""

    return load_yaml(path)


def normalize_sample_annotation(
    sample: dict[str, Any] | Any,
    config: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Normalize one dataset sample and return warning records.

    Samples without an annotation are skipped from normalized label outputs but
    still produce a warning for audit coverage.
    """

    sample_dict = sample_to_dict(sample)
    sample_id = str(sample_dict.get("sample_id", ""))
    dataset_name = str(sample_dict.get("dataset_name", ""))
    warnings: list[dict[str, Any]] = []
    annotation = unwrap_annotation(sample_dict.get("annotation"))
    if not annotation:
        return None, [
            _warning(
                sample_id,
                dataset_name,
                "missing_annotation",
                "annotation",
                None,
                "Sample has no annotation payload.",
            )
        ]

    surface = get_block(annotation, "surface_content")
    target = get_block(annotation, "target")
    intent = get_block(annotation, "intent")
    tactic = get_block(annotation, "tactic")
    evidence = get_block(annotation, "evidence")
    quality = get_block(annotation, "quality")

    harmfulness = map_raw_harmfulness(sample_dict.get("raw_label"), dataset_name, config)
    if harmfulness == "unknown":
        warnings.append(
            _warning(
                sample_id,
                dataset_name,
                "unknown_harmfulness",
                "raw_label",
                sample_dict.get("raw_label"),
                "Raw label could not be mapped by dataset-specific mapping.",
            )
        )

    target_presence, unknown = normalize_label(target.get("target_presence"), "target_presence", config)
    if unknown:
        warnings.append(_warning(sample_id, dataset_name, "unknown_target_presence", "target.target_presence", target.get("target_presence")))

    target_granularity, unknown = normalize_label(target.get("target_granularity"), "target_granularity", config)
    if unknown and not is_missing(target.get("target_granularity")):
        warnings.append(_warning(sample_id, dataset_name, "unknown_target_granularity", "target.target_granularity", target.get("target_granularity")))

    protected_attribute, unknown_attributes = normalize_label_list(
        target.get("protected_attribute"),
        "protected_attribute",
        config,
    )
    for value in unknown_attributes:
        warnings.append(
            _warning(
                sample_id,
                dataset_name,
                "unknown_protected_attribute",
                "target.protected_attribute",
                value,
            )
        )

    intent_primary, unknown = normalize_label(intent.get("intent_primary"), "intent_primary", config)
    if unknown:
        warnings.append(_warning(sample_id, dataset_name, "unknown_intent", "intent.intent_primary", intent.get("intent_primary")))

    secondary_intent = _clean_string_list(intent.get("secondary_intent"))
    stance, unknown = normalize_label(intent.get("stance"), "stance", config)
    if unknown and not is_missing(intent.get("stance")):
        warnings.append(_warning(sample_id, dataset_name, "unknown_stance", "intent.stance", intent.get("stance")))

    tactic_rhetorical, unknown_tactics = normalize_label_list(tactic.get("tactic_rhetorical"), "tactic_rhetorical", config)
    for value in unknown_tactics:
        warnings.append(_warning(sample_id, dataset_name, "unknown_tactic", "tactic.tactic_rhetorical", value))

    tactic_relation, unknown = normalize_label(tactic.get("tactic_multimodal_relation"), "tactic_multimodal_relation", config)
    if unknown and not is_missing(tactic.get("tactic_multimodal_relation")):
        warnings.append(_warning(sample_id, dataset_name, "unknown_tactic_relation", "tactic.tactic_multimodal_relation", tactic.get("tactic_multimodal_relation")))

    confidence, unknown = normalize_label(quality.get("confidence"), "confidence", config)
    if unknown and not is_missing(quality.get("confidence")):
        warnings.append(_warning(sample_id, dataset_name, "unknown_confidence", "quality.confidence", quality.get("confidence")))

    labels = {
        "harmfulness": harmfulness,
        "target_presence": target_presence,
        "target_granularity": target_granularity,
        "protected_attribute": protected_attribute,
        "intent_primary": intent_primary,
        "secondary_intent": secondary_intent,
        "stance": stance,
        "background_knowledge_needed": parse_bool(intent.get("background_knowledge_needed"), default=False),
        "tactic_rhetorical": tactic_rhetorical,
        "tactic_multimodal_relation": tactic_relation,
        "not_sure": parse_bool(quality.get("not_sure"), default=False),
        "confidence": confidence,
        "confidence_score": confidence_score(confidence, config),
    }

    evidence_text = {
        "visual_scene_summary": _string(surface.get("visual_scene_summary")),
        "surface_multimodal_relation": _string(surface.get("multimodal_relation")),
        "target_text_span": _string(target.get("target_text_span")),
        "target_visual_cue": _string(target.get("target_visual_cue")),
        "intent_free_text": _string(intent.get("intent_free_text")),
        "background_knowledge_text": _string(intent.get("background_knowledge_text")),
        "tactic_text_span": _string(tactic.get("evidence_text_span")),
        "tactic_image_region_description": _string(tactic.get("evidence_image_region_description")),
        "key_text_evidence": _string(evidence.get("key_text_evidence")),
        "key_visual_evidence": _string(evidence.get("key_visual_evidence")),
        "key_cross_modal_evidence": _string(evidence.get("key_cross_modal_evidence")),
    }

    audit_flags = compute_audit_flags(labels, sample_dict)
    row = {
        "sample_id": sample_id,
        "dataset_name": dataset_name,
        "image_path": sample_dict.get("image_path"),
        "ocr_text_full": sample_dict.get("ocr_text_full") or surface.get("ocr_text_full") or "",
        "raw_label": sample_dict.get("raw_label"),
        "labels": labels,
        "evidence_text": evidence_text,
        "source_annotation": {
            "has_annotation": True,
            "annotation_schema_version": "v1_silver",
            "raw_annotation_keys": raw_annotation_keys(annotation),
        },
        "audit_flags": audit_flags,
    }
    return row, warnings


def compute_audit_flags(labels: dict[str, Any], sample: dict[str, Any] | None = None) -> list[str]:
    """Compute conservative audit flags from normalized labels."""

    flags: list[str] = []
    harmfulness = labels.get("harmfulness")
    target_presence = labels.get("target_presence")
    intent = labels.get("intent_primary")
    stance = labels.get("stance")
    tactics = set(labels.get("tactic_rhetorical") or [])
    if labels.get("background_knowledge_needed"):
        flags.append("background_knowledge_needed")
    if labels.get("not_sure"):
        flags.append("not_sure_yes")
    if labels.get("confidence") == "low":
        flags.append("low_confidence")
    if harmfulness == "unknown":
        flags.append("unknown_harmfulness")
    if target_presence == "unknown":
        flags.append("unknown_target_presence")
    if intent == "unknown":
        flags.append("unknown_intent")
    if "unknown" in tactics:
        flags.append("unknown_tactic")
    hostile_intents = {"ridicule_mockery", "denigration_insult", "harassment_threat", "mobilization_incitement", "misinformation_deception"}
    hostile_tactics = {"slur", "smear", "dehumanization", "objectification", "exclusion", "dog_whistle"}
    if harmfulness == "non_harmful" and (stance == "hostile" or intent in hostile_intents or tactics & hostile_tactics):
        flags.append("raw_label_non_harmful_but_hostile")
    if harmfulness == "non_harmful" and target_presence in {"explicit", "implicit"}:
        flags.append("raw_label_non_harmful_but_target_present")
    if harmfulness == "harmful" and target_presence == "none":
        flags.append("raw_label_harmful_but_target_none")
    if harmfulness == "harmful" and intent in {"entertainment", "self_expression"}:
        flags.append("raw_label_harmful_but_entertainment_or_self_expression")
    if harmfulness == "harmful" and stance == "neutral":
        flags.append("raw_label_harmful_but_neutral_stance")
    return sorted(dict.fromkeys(flags))


def build_normalized_dataset(
    dataset_names: list[str],
    dataset_root: str | Path,
    annotation_root: str | Path,
    config: dict[str, Any],
    limit: int | None = None,
    disable_tqdm: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load samples through MemeDataset and normalize their annotations."""

    dataset = MemeDataset(
        dataset_root=dataset_root,
        annotation_root=annotation_root,
        dataset_names=dataset_names,
        keep_missing_images=True,
        limit=limit,
    )
    normalized: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for sample in progress_iter(dataset, desc="normalize annotations", leave=False, disable=disable_tqdm):
        row, row_warnings = normalize_sample_annotation(sample, config)
        warnings.extend(row_warnings)
        if row is not None:
            normalized.append(row)
    return normalized, warnings


def write_normalized_outputs(
    rows: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    normalized_root: str | Path = "dataset/annotation_normalized",
    *,
    clean_confidence: set[str] | None = None,
    include_not_sure: bool = False,
) -> dict[str, Path]:
    """Write per-dataset and aggregate normalized JSONL outputs."""

    clean_confidence = clean_confidence or {"high", "medium"}
    root = Path(normalized_root)
    outputs: dict[str, Path] = {}
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_warnings: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[str(row.get("dataset_name"))].append(row)
        grouped_rows["all"].append(row)
    for warning in warnings:
        grouped_warnings[str(warning.get("dataset_name") or "all")].append(warning)
        grouped_warnings["all"].append(warning)

    for dataset_name, dataset_rows in grouped_rows.items():
        out_dir = root / dataset_name
        clean_rows = [row for row in dataset_rows if is_clean_row(row, clean_confidence, include_not_sure)]
        labels_path = out_dir / "normalized_labels.jsonl"
        clean_path = out_dir / "normalized_clean.jsonl"
        write_jsonl(labels_path, dataset_rows, compact_tensors=False)
        write_jsonl(clean_path, clean_rows, compact_tensors=False)
        outputs[f"{dataset_name}_labels"] = labels_path
        outputs[f"{dataset_name}_clean"] = clean_path

    for dataset_name, dataset_warnings in grouped_warnings.items():
        path = root / dataset_name / "normalization_warnings.jsonl"
        write_jsonl(path, dataset_warnings, compact_tensors=False)
        outputs[f"{dataset_name}_warnings"] = path
    return outputs


def is_clean_row(row: dict[str, Any], clean_confidence: set[str], include_not_sure: bool = False) -> bool:
    """Return whether a normalized row belongs in normalized_clean.jsonl."""

    labels = row.get("labels", {})
    if labels.get("confidence") not in clean_confidence:
        return False
    if not include_not_sure and labels.get("not_sure"):
        return False
    return True


def resolve_dataset_names(requested: str | list[str] | None, config: dict[str, Any]) -> list[str]:
    """Resolve CLI dataset argument into canonical dataset names."""

    datasets = list(config.get("datasets") or DEFAULT_DATASETS)
    if requested is None:
        return datasets
    if isinstance(requested, str):
        requested_values = [requested]
    else:
        requested_values = requested
    if "all" in requested_values:
        return datasets
    return requested_values


def summarize_normalization(rows: list[dict[str, Any]], warnings: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a compact summary suitable for stdout."""

    by_dataset: dict[str, int] = defaultdict(int)
    for row in rows:
        by_dataset[str(row.get("dataset_name"))] += 1
    warning_counts: dict[str, int] = defaultdict(int)
    for warning in warnings:
        warning_counts[str(warning.get("type"))] += 1
    return {
        "normalized_rows": len(rows),
        "warnings": len(warnings),
        "rows_by_dataset": dict(sorted(by_dataset.items())),
        "warning_counts": dict(sorted(warning_counts.items())),
    }


def _warning(
    sample_id: str,
    dataset_name: str,
    warning_type: str,
    field: str,
    value: Any,
    message: str | None = None,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "dataset_name": dataset_name,
        "type": warning_type,
        "field": field,
        "value": value,
        "message": message or warning_type,
    }


def _string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return "; ".join(str(item) for item in value if item is not None)
    return str(value)


def _clean_string_list(value: Any) -> list[str]:
    from utils.annotation_utils import as_list

    cleaned = []
    for item in as_list(value):
        text = str(item).strip()
        if text and text.lower() not in {"none", "null", "unknown"} and text not in cleaned:
            cleaned.append(text)
    return cleaned
