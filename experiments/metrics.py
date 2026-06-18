"""Metrics for harmfulness classification experiments."""

from __future__ import annotations

from math import isnan
from typing import Any, Iterable


def compute_harmfulness_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    prob_harmful: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Compute binary harmfulness metrics with safe edge-case handling."""

    true = [int(y) for y in y_true]
    pred = [int(y) for y in y_pred]
    probs = [float(p) for p in prob_harmful] if prob_harmful is not None else []
    if not true or not pred:
        return _empty_metrics()

    tn = sum(1 for y, p in zip(true, pred) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(true, pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(true, pred) if y == 1 and p == 0)
    tp = sum(1 for y, p in zip(true, pred) if y == 1 and p == 1)

    accuracy = (tp + tn) / max(1, len(true))
    precision_pos = tp / max(1, tp + fp)
    recall_pos = tp / max(1, tp + fn)
    f1_pos = _f1(precision_pos, recall_pos)

    precision_neg = tn / max(1, tn + fn)
    recall_neg = tn / max(1, tn + fp)
    f1_neg = _f1(precision_neg, recall_neg)

    support_pos = sum(1 for y in true if y == 1)
    support_neg = sum(1 for y in true if y == 0)
    macro_f1 = (f1_pos + f1_neg) / 2.0
    weighted_f1 = (f1_pos * support_pos + f1_neg * support_neg) / max(1, len(true))
    roc_auc = _roc_auc(true, probs) if probs else None

    return {
        "accuracy": accuracy,
        "precision": precision_pos,
        "recall": recall_pos,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "roc_auc": roc_auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def metrics_to_csv_row(dataset: str, model: str, seed: int | str, metrics: dict[str, Any]) -> dict[str, Any]:
    """Normalize metrics into the main performance table row schema."""

    return {
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
        "roc_auc": metrics.get("roc_auc"),
        "tn": metrics.get("tn"),
        "fp": metrics.get("fp"),
        "fn": metrics.get("fn"),
        "tp": metrics.get("tp"),
    }


def summarize_mean_std(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate metrics by dataset/model across seeds."""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["dataset"]), str(row["model"])), []).append(row)

    metric_names = ["accuracy", "precision", "recall", "macro_f1", "weighted_f1", "roc_auc"]
    count_names = ["tn", "fp", "fn", "tp"]
    output: list[dict[str, Any]] = []
    for (dataset, model), group in sorted(groups.items()):
        summary: dict[str, Any] = {"dataset": dataset, "model": model, "num_seeds": len(group)}
        for name in metric_names:
            values = [_to_float(row.get(name)) for row in group]
            values = [value for value in values if value is not None and not isnan(value)]
            summary[f"{name}_mean"] = sum(values) / len(values) if values else None
            summary[f"{name}_std"] = _std(values) if len(values) > 1 else 0.0 if values else None
        for name in count_names:
            values = [_to_float(row.get(name)) for row in group]
            values = [value for value in values if value is not None and not isnan(value)]
            summary[f"{name}_mean"] = sum(values) / len(values) if values else None
        output.append(summary)
    return output


def _empty_metrics() -> dict[str, Any]:
    return {
        "accuracy": None,
        "precision": None,
        "recall": None,
        "macro_f1": None,
        "weighted_f1": None,
        "roc_auc": None,
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "tp": 0,
    }


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _roc_auc(y_true: list[int], scores: list[float]) -> float | None:
    """Compute ROC-AUC from ranks; returns None if only one class exists."""

    if len(y_true) != len(scores) or not y_true:
        return None
    positives = [score for y, score in zip(y_true, scores) if y == 1]
    negatives = [score for y, score in zip(y_true, scores) if y == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = len(positives) * len(negatives)
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total


def _std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
