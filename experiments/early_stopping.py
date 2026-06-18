"""Reusable early stopping and checkpoint helpers."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch

from utils.io import write_json


class EarlyStopping:
    """Track validation improvements and patience-based stopping."""

    def __init__(self, patience: int = 3, min_delta: float = 0.0, mode: str = "max") -> None:
        if mode not in {"max", "min"}:
            raise ValueError("early_stop_mode must be 'max' or 'min'")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best_metric: float | None = None
        self.best_epoch: int | None = None
        self.counter = 0
        self.stopped_early = False

    @property
    def enabled(self) -> bool:
        """Whether early stopping is active."""

        return self.patience > 0

    def step(self, value: float | int | None, epoch: int, active: bool = True) -> dict[str, Any]:
        """Update state and return status for the current epoch."""

        metric = _to_float(value)
        improved = False
        if active and metric is not None:
            improved = self._is_improvement(metric)
            if improved:
                self.best_metric = metric
                self.best_epoch = epoch
                self.counter = 0
            else:
                self.counter += 1
        elif active:
            self.counter += 1

        if active and self.enabled and self.counter >= self.patience:
            self.stopped_early = True

        return {
            "is_best": improved,
            "best_metric_so_far": self.best_metric,
            "best_epoch": self.best_epoch,
            "patience_counter": self.counter,
            "stopped_early": self.stopped_early,
        }

    def _is_improvement(self, metric: float) -> bool:
        if self.best_metric is None:
            return True
        if self.mode == "max":
            return metric >= self.best_metric + self.min_delta
        return metric <= self.best_metric - self.min_delta


def metric_from_validation(
    validation_metrics: dict[str, Any],
    metric_name: str,
    structured_score: bool = False,
) -> float | None:
    """Extract an early-stop metric from validation metrics."""

    if structured_score or metric_name in {"val_structured_score", "structured_score"}:
        return structured_validation_score(validation_metrics)
    candidates = [metric_name]
    if metric_name.startswith("val_"):
        candidates.append(metric_name[4:])
    else:
        candidates.append(f"val_{metric_name}")
    if metric_name in {"val_macro_f1", "macro_f1"}:
        candidates.extend(["harmfulness_macro_f1", "val_harmfulness_macro_f1"])
    for key in candidates:
        value = _to_float(validation_metrics.get(key))
        if value is not None:
            return value
    return None


def structured_validation_score(validation_metrics: dict[str, Any]) -> float | None:
    """Mean harmfulness/target/intent/tactic score, skipping missing components."""

    values: list[float] = []
    harmfulness = _to_float(validation_metrics.get("harmfulness_macro_f1"))
    if harmfulness is None:
        harmfulness = _to_float(validation_metrics.get("macro_f1"))
    if harmfulness is not None:
        values.append(harmfulness)
    keys = ["target_granularity_macro_f1", "intent_primary_macro_f1", "tactic_multimodal_relation_macro_f1"]
    for key in keys:
        value = _to_float(validation_metrics.get(key))
        if value is not None:
            values.append(value)
    return sum(values) / len(values) if values else None


def prefix_metrics(metrics: dict[str, Any], prefix: str = "val_") -> dict[str, Any]:
    """Return metrics with a prefix for log rows."""

    return {f"{prefix}{key}": value for key, value in metrics.items()}


def save_training_log(output_dir: str | Path, rows: list[dict[str, Any]]) -> None:
    """Save training log as JSON and CSV."""

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    write_json(path / "training_log.json", rows)
    columns = sorted({key for row in rows for key in row}) if rows else ["epoch"]
    with (path / "training_log.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    path: str | Path,
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    best_metric: float | None,
    config: Any,
    seed: int,
    dataset: str,
    model_name: str,
    scheduler: Any | None = None,
) -> Path:
    """Save a resume/evaluation-friendly checkpoint."""

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "best_metric": best_metric,
        "config": asdict(config) if is_dataclass(config) else config,
        "seed": seed,
        "dataset": dataset,
        "model_name": model_name,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(metric) else metric
