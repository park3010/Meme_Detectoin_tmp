"""Paper-ready CSV table export helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


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
