"""Dataset statistics and annotation coverage reporting."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from dataset import MemeDataset
from experiments.splits import label_to_int, normalize_dataset_names
from utils.io import write_json


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
