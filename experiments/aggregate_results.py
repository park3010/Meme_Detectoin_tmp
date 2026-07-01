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
