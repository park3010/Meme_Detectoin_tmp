"""Build normalized silver labels from LLM-generated annotation JSONL files."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dataset import MemeDataset
from experiments.progress import progress_iter
from experiments.splits import label_to_int, normalize_dataset_names
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
from utils.io import load_yaml, write_json, write_jsonl


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


# =============================================================================
# Annotation audit
# =============================================================================

LABEL_FIELDS = [
    "harmfulness",
    "target_presence",
    "target_granularity",
    "protected_attribute",
    "intent_primary",
    "stance",
    "background_knowledge_needed",
    "tactic_rhetorical",
    "tactic_multimodal_relation",
    "not_sure",
    "confidence",
]

CROSSTABS = [
    ("harmfulness", "target_presence"),
    ("harmfulness", "target_granularity"),
    ("harmfulness", "intent_primary"),
    ("harmfulness", "stance"),
    ("harmfulness", "background_knowledge_needed"),
    ("harmfulness", "tactic_rhetorical"),
    ("not_sure", "confidence"),
]


def run_annotation_audit(
    dataset_names: list[str],
    dataset_root: str | Path,
    annotation_root: str | Path,
    config: dict[str, Any],
    audit_root: str | Path = "result/annotation_audit",
    limit: int | None = None,
    max_examples_per_flag: int = 50,
    disable_tqdm: bool = False,
) -> dict[str, Any]:
    """Load samples, normalize annotations in memory, and write audit outputs."""

    dataset = MemeDataset(
        dataset_root=dataset_root,
        annotation_root=annotation_root,
        dataset_names=dataset_names,
        keep_missing_images=True,
        limit=limit,
    )
    rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    samples = []
    for sample in progress_iter(dataset, desc="audit annotations", leave=False, disable=disable_tqdm):
        samples.append(sample)
        row, row_warnings = normalize_sample_annotation(sample, config)
        warnings.extend(row_warnings)
        if row:
            rows.append(row)

    outputs: dict[str, Any] = {}
    for dataset_name in [*dataset_names, "all"]:
        dataset_rows = rows if dataset_name == "all" else [row for row in rows if row.get("dataset_name") == dataset_name]
        dataset_warnings = warnings if dataset_name == "all" else [w for w in warnings if w.get("dataset_name") == dataset_name]
        dataset_samples = samples if dataset_name == "all" else [s for s in samples if s.get("dataset_name") == dataset_name]
        audit = compute_annotation_audit(dataset_rows, dataset_warnings, dataset_samples)
        outputs[dataset_name] = write_audit_outputs(
            dataset_name,
            audit,
            dataset_rows,
            dataset_warnings,
            audit_root,
            max_examples_per_flag=max_examples_per_flag,
        )
    return {
        "rows": len(rows),
        "warnings": len(warnings),
        "datasets": dataset_names,
        "outputs": outputs,
    }


def compute_annotation_audit(
    normalized_rows: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute coverage, label distributions, crosstabs, and flag counts."""

    samples = samples or []
    sample_keys = {
    (str(row.get("dataset_name")), str(row.get("sample_id")))
    for row in normalized_rows
    }
    missing_annotation_keys = {
        (str(w.get("dataset_name")), str(w.get("sample_id")))
        for w in warnings
        if w.get("type") == "missing_annotation"
    }
    total_loaded = len(samples) if samples else len(sample_keys | missing_annotation_keys)
    with_annotation = len(sample_keys)
    without_annotation = len(missing_annotation_keys)
    missing_images = sum(1 for sample in samples if not _image_exists(sample)) if samples else sum(1 for row in normalized_rows if not row.get("image_path"))
    empty_ocr = sum(1 for sample in samples if not str(sample.get("ocr_text_full", "")).strip()) if samples else sum(1 for row in normalized_rows if not str(row.get("ocr_text_full", "")).strip())

    raw_labels = Counter(str(row.get("raw_label")) for row in normalized_rows)
    distributions = label_distributions(normalized_rows)
    crosstabs = {f"{left}_x_{right}": crosstab(normalized_rows, left, right) for left, right in CROSSTABS}
    flag_counts = Counter(flag for row in normalized_rows for flag in row.get("audit_flags", []))
    warning_counts = Counter(str(w.get("type")) for w in warnings)
    return {
        "summary": {
            "total_loaded_samples": total_loaded,
            "samples_with_annotation": with_annotation,
            "samples_without_annotation": without_annotation,
            "missing_images": missing_images,
            "empty_ocr_strings": empty_ocr,
            "annotation_coverage_ratio": with_annotation / total_loaded if total_loaded else 0.0,
            "normalization_warning_count": len(warnings),
            "flagged_sample_count": sum(1 for row in normalized_rows if row.get("audit_flags")),
        },
        "raw_label_distribution": dict(sorted(raw_labels.items())),
        "label_distributions": distributions,
        "crosstabs": crosstabs,
        "audit_flag_counts": dict(sorted(flag_counts.items())),
        "warning_counts": dict(sorted(warning_counts.items())),
    }


def label_distributions(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Count normalized label values, expanding list fields."""

    distributions: dict[str, Counter[str]] = {field: Counter() for field in LABEL_FIELDS}
    for row in rows:
        labels = row.get("labels", {})
        for field in LABEL_FIELDS:
            value = labels.get(field)
            if isinstance(value, list):
                if not value:
                    distributions[field][""] += 1
                for item in value:
                    distributions[field][str(item)] += 1
            else:
                distributions[field][str(value)] += 1
    return {field: dict(sorted(counter.items())) for field, counter in distributions.items()}


def crosstab(rows: list[dict[str, Any]], left: str, right: str) -> list[dict[str, Any]]:
    """Return row-count crosstab records, expanding multi-label sides."""

    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        labels = row.get("labels", {})
        for left_value in _values(labels.get(left)):
            for right_value in _values(labels.get(right)):
                counts[(left_value, right_value)] += 1
    return [
        {left: left_value, right: right_value, "count": count}
        for (left_value, right_value), count in sorted(counts.items())
    ]


def write_audit_outputs(
    dataset_name: str,
    audit: dict[str, Any],
    rows: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    audit_root: str | Path,
    *,
    max_examples_per_flag: int = 50,
) -> dict[str, str]:
    """Write audit JSON/CSV/JSONL artifacts for one dataset grouping."""

    out_dir = Path(audit_root) / dataset_name
    crosstab_dir = out_dir / "crosstabs"
    write_json(out_dir / "audit_summary.json", audit["summary"] | {
        "raw_label_distribution": audit["raw_label_distribution"],
        "audit_flag_counts": audit["audit_flag_counts"],
        "warning_counts": audit["warning_counts"],
    })
    write_json(out_dir / "label_distributions.json", audit["label_distributions"])
    _write_distribution_csv(out_dir / "label_distributions.csv", audit["label_distributions"])
    for name, records in audit["crosstabs"].items():
        _write_records_csv(crosstab_dir / f"{name}.csv", records)
    write_jsonl(out_dir / "flagged_examples.jsonl", _flagged_examples(rows, max_examples_per_flag), compact_tensors=False)
    write_jsonl(out_dir / "normalization_warnings.jsonl", warnings, compact_tensors=False)
    return {
        "summary": str(out_dir / "audit_summary.json"),
        "distributions": str(out_dir / "label_distributions.csv"),
        "flagged_examples": str(out_dir / "flagged_examples.jsonl"),
        "warnings": str(out_dir / "normalization_warnings.jsonl"),
    }


def format_audit_summary(result: dict[str, Any]) -> str:
    """Create a compact stdout summary for CLI scripts."""

    lines = [
        f"Audited datasets: {', '.join(result.get('datasets', []))}",
        f"Normalized annotated rows: {result.get('rows', 0)}",
        f"Warnings: {result.get('warnings', 0)}",
    ]
    for dataset_name, paths in sorted(result.get("outputs", {}).items()):
        lines.append(f"- {dataset_name}: {paths.get('summary')}")
    return "\n".join(lines)


def _flagged_examples(rows: list[dict[str, Any]], max_examples_per_flag: int) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for row in rows:
        flags = list(row.get("audit_flags", []))
        if not flags:
            continue
        if max_examples_per_flag > 0 and not any(counts[flag] < max_examples_per_flag for flag in flags):
            continue
        for flag in flags:
            counts[flag] += 1
        selected.append(
            {
                "sample_id": row.get("sample_id"),
                "dataset_name": row.get("dataset_name"),
                "image_path": row.get("image_path"),
                "ocr_text_full": row.get("ocr_text_full"),
                "raw_label": row.get("raw_label"),
                "labels": row.get("labels", {}),
                "audit_flags": flags,
                "evidence_text": row.get("evidence_text", {}),
            }
        )
    return selected


def _write_distribution_csv(path: Path, distributions: dict[str, dict[str, int]]) -> None:
    rows = [
        {"field": field, "label": label, "count": count}
        for field, counter in sorted(distributions.items())
        for label, count in sorted(counter.items())
    ]
    _write_records_csv(path, rows, columns=["field", "label", "count"])


def _write_records_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = columns or sorted({key for row in rows for key in row}) or ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value] or [""]
    return [str(value)]


def _image_exists(sample: dict[str, Any]) -> bool:
    metadata = sample.get("metadata", {}) if isinstance(sample.get("metadata"), dict) else {}
    if "image_exists" in metadata:
        return bool(metadata["image_exists"])
    image_path = sample.get("image_path")
    return bool(image_path and Path(str(image_path)).exists())


# =============================================================================
# Dataset statistics
# =============================================================================

STAT_COLUMNS = [
    "dataset_name",
    "total_samples",
    "harmful_count",
    "non_harmful_count",
    "harmful_ratio",
    "image_missing_count",
    "ocr_missing_count",
    "annotation_count",
    "target_label_coverage",
    "intent_label_coverage",
    "tactic_label_coverage",
    "knowledge_needed_count",
    "knowledge_needed_ratio",
]


def compute_dataset_statistics(
    dataset_root: str | Path = "dataset/source",
    annotation_root: str | Path = "dataset/annotation",
    dataset_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Compute paper Table 1 style statistics."""

    names = normalize_dataset_names(dataset_names)
    dataset = MemeDataset(dataset_root=dataset_root, annotation_root=annotation_root, dataset_names=names, keep_missing_images=True)
    rows: list[dict[str, Any]] = []
    for dataset_name in sorted({sample.dataset_name for sample in dataset.samples}):
        samples = [sample for sample in dataset.samples if sample.dataset_name == dataset_name]
        harmful = sum(1 for sample in samples if label_to_int(sample.raw_label) == 1)
        non_harmful = sum(1 for sample in samples if label_to_int(sample.raw_label) == 0)
        annotations = [sample.annotation for sample in samples if sample.annotation]
        knowledge_needed = sum(1 for annotation in annotations if _knowledge_needed(annotation))
        rows.append(
            {
                "dataset_name": dataset_name,
                "total_samples": len(samples),
                "harmful_count": harmful,
                "non_harmful_count": non_harmful,
                "harmful_ratio": harmful / max(1, harmful + non_harmful),
                "image_missing_count": sum(1 for sample in samples if not sample.image_path or not Path(sample.image_path).exists()),
                "ocr_missing_count": sum(1 for sample in samples if not sample.ocr_text_full.strip()),
                "annotation_count": len(annotations),
                "target_label_coverage": _coverage(annotations, ["target", "target_granularity"]),
                "intent_label_coverage": _coverage(annotations, ["intent", "intent_primary"]),
                "tactic_label_coverage": _coverage(annotations, ["tactic", "tactic_rhetorical"]),
                "knowledge_needed_count": knowledge_needed,
                "knowledge_needed_ratio": knowledge_needed / max(1, len(annotations)),
            }
        )
    return rows


def save_dataset_statistics(rows: list[dict[str, Any]], output_root: str | Path = "result/dataset_stats") -> tuple[Path, Path]:
    """Save dataset statistics as JSON and CSV."""

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "dataset_statistics.json"
    csv_path = output_dir / "dataset_statistics.csv"
    write_json(json_path, rows)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STAT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in STAT_COLUMNS})
    return json_path, csv_path


def format_statistics_summary(rows: list[dict[str, Any]]) -> str:
    """Return a readable stdout summary."""

    lines = ["Dataset statistics:"]
    for row in rows:
        lines.append(
            (
                f"- {row['dataset_name']}: total={row['total_samples']} "
                f"harmful={row['harmful_count']} non_harmful={row['non_harmful_count']} "
                f"harmful_ratio={row['harmful_ratio']:.3f} annotations={row['annotation_count']}"
            )
        )
    return "\n".join(lines)


def _coverage(annotations: list[dict[str, Any]], path: list[str]) -> float:
    if not annotations:
        return 0.0
    count = 0
    for annotation in annotations:
        value: Any = annotation
        for key in path:
            value = value.get(key, {}) if isinstance(value, dict) else {}
        if value not in ({}, [], "", None, ["none"]):
            count += 1
    return count / len(annotations)


def _knowledge_needed(annotation: dict[str, Any]) -> bool:
    intent = annotation.get("intent", {}) if isinstance(annotation, dict) else {}
    if isinstance(intent, dict) and bool(intent.get("background_knowledge_needed")):
        return True
    text = str(intent.get("background_knowledge_text", "") if isinstance(intent, dict) else "")
    return bool(text.strip())
