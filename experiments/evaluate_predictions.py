"""Evaluate saved prediction JSONL files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.metrics import compute_harmfulness_metrics
from utils.io import read_jsonl, write_json


def evaluate_prediction_file(path: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    """Evaluate one prediction JSONL file and optionally save metrics JSON."""

    records = read_jsonl(path)
    metrics = compute_harmfulness_metrics(
        [int(record["gold_label"]) for record in records],
        [int(record["pred_label"]) for record in records],
        [float(record["prob_harmful"]) for record in records],
    )
    if output_path is not None:
        write_json(output_path, metrics)
    return metrics
