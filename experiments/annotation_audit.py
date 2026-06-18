"""Audit normalized silver labels and annotation coverage."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dataset import MemeDataset
from experiments.annotation_normalization import normalize_sample_annotation
from experiments.progress import progress_iter
from utils.io import write_json, write_jsonl


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
