"""Aggregate baseline metrics into paper-table CSV files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from experiments.evaluation import metrics_to_csv_row, summarize_mean_std


MAIN_COLUMNS = ["dataset", "model", "seed", "accuracy", "precision", "recall", "macro_f1", "weighted_f1", "roc_auc", "tn", "fp", "fn", "tp"]


def collect_metric_rows(predictions_root: str | Path = "result/predictions") -> list[dict[str, Any]]:
    """Scan result/predictions/{dataset}/{model}/{seed}/metrics.json."""

    rows: list[dict[str, Any]] = []
    root = Path(predictions_root)
    for path in sorted(root.glob("*/*/*/metrics.json")):
        dataset, model, seed = path.parts[-4], path.parts[-3], path.parts[-2]
        with path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        rows.append(metrics_to_csv_row(dataset, model, seed, metrics))
    return rows


def write_aggregate_tables(
    predictions_root: str | Path = "result/predictions",
    output_root: str | Path = "result/metrics",
) -> tuple[Path, Path]:
    """Write per-seed and mean/std performance CSV files."""

    rows = collect_metric_rows(predictions_root)
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    main_path = output_dir / "main_performance.csv"
    mean_std_path = output_dir / "main_performance_mean_std.csv"
    _write_csv(main_path, rows, MAIN_COLUMNS)
    mean_std_rows = summarize_mean_std(rows)
    if mean_std_rows:
        columns = list(mean_std_rows[0].keys())
    else:
        columns = ["dataset", "model", "num_seeds"]
    _write_csv(mean_std_path, mean_std_rows, columns)
    return main_path, mean_std_path


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


# =============================================================================
# Intermediate result manifest export
# =============================================================================

def export_intermediate_manifest(
    result_root: str | Path = "result",
    output: str | Path = "result/intermediate_manifest.json",
) -> dict[str, Any]:
    """Export a compact manifest of files currently available under result_root."""

    root = Path(result_root)
    files = [
        {"path": str(path), "bytes": path.stat().st_size}
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    manifest = {"result_root": str(root), "files": files}
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


# =============================================================================
# Paper table export
# =============================================================================

TABLE_SOURCES = {
    "table1_dataset_statistics.csv": ["dataset_stats/dataset_statistics.csv"],
    "table2_main_performance.csv": ["metrics/main_performance_mean_std.csv", "metrics/main_performance.csv"],
    "table3_structured_interpretation.csv": ["metrics/structured_interpretation_mean_std.csv", "metrics/structured_interpretation.csv"],
    "table4_stagewise_ablation.csv": ["metrics/ablation.csv"],
    "table5_knowledge_comparison.csv": ["metrics/knowledge_comparison.csv"],
    "table6_verifier_evaluation.csv": ["metrics/verifier_eval.csv"],
    "table7_cross_domain.csv": ["metrics/cross_domain.csv"],
    "table8_runtime_cost.csv": ["metrics/runtime_cost.csv"],
}


def export_paper_tables(result_root: str = "result", output_root: str | None = None) -> list[Path]:
    """Export normalized paper table CSVs from existing metric artifacts."""

    root = Path(result_root)
    out = Path(output_root) if output_root else root / "paper_tables"
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for table_name, candidates in TABLE_SOURCES.items():
        source = next((root / rel for rel in candidates if (root / rel).exists()), None)
        destination = out / table_name
        if source is None:
            _write_missing(destination, f"Missing source: one of {', '.join(candidates)}")
        else:
            _normalize_csv(source, destination)
        written.append(destination)
    return written


def _normalize_csv(source: Path, destination: Path) -> None:
    rows = _read_csv(source)
    if not rows:
        _write_missing(destination, f"Empty source: {source}")
        return
    rows = [_with_notes(row) for row in rows]
    columns = list(rows[0])
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _with_notes(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    missing = [key for key, value in out.items() if value in {None, "", "nan", "NaN"}]
    out["notes"] = "missing: " + ", ".join(missing) if missing else ""
    return out


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_missing(path: Path, note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["notes"])
        writer.writeheader()
        writer.writerow({"notes": note})
