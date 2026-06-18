"""Small evaluation helpers for smoke tests and result inspection."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable


def label_distribution(labels: Iterable[object]) -> dict[str, int]:
    """Return a string-keyed label distribution."""

    return dict(Counter(str(label) for label in labels))


def binary_classification_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> dict[str, float]:
    """Compute binary accuracy, precision, recall, and F1."""

    true = list(y_true)
    pred = list(y_pred)
    tp = sum(1 for y, p in zip(true, pred) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(true, pred) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(true, pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(true, pred) if y == 1 and p == 0)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {
        "accuracy": (tp + tn) / max(1, len(true)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def confusion_matrix_summary(y_true: Iterable[str], y_pred: Iterable[str]) -> dict[str, dict[str, int]]:
    """Return a nested confusion matrix summary."""

    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for gold, pred in zip(y_true, y_pred):
        matrix[str(gold)][str(pred)] += 1
    return {gold: dict(row) for gold, row in matrix.items()}


def multiclass_metrics(y_true: Iterable[str], y_pred: Iterable[str]) -> dict[str, object]:
    """Compute simple multiclass accuracy and confusion matrix."""

    true = [str(item) for item in y_true]
    pred = [str(item) for item in y_pred]
    accuracy = sum(1 for y, p in zip(true, pred) if y == p) / max(1, len(true))
    return {"accuracy": accuracy, "confusion": confusion_matrix_summary(true, pred)}


def evidence_precision_recall_at_k(predicted: list[str], gold: Iterable[str], k: int = 3) -> dict[str, float]:
    """Compute evidence attribution precision@k and recall@k."""

    pred_at_k = predicted[:k]
    gold_set = set(str(item) for item in gold)
    if not gold_set:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0}
    hits = sum(1 for item in pred_at_k if item in gold_set)
    return {"precision_at_k": hits / max(1, len(pred_at_k)), "recall_at_k": hits / len(gold_set)}
