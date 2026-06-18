"""Statistical significance reporting for multi-seed metrics."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from statistics import mean, stdev
from typing import Any


COMPARISONS = [
    ("ours_full", "best_simple_baseline"),
    ("ours_full", "ablation_w_o_retrieval"),
    ("ours_full", "ablation_w_o_support_verifier"),
    ("ours_full", "ablation_w_o_task_aware_gate"),
    ("knowledge_verified", "knowledge_retrieved_only"),
]

SIMPLE_BASELINES = ["image_only_clip", "text_only_encoder", "clip_text_concat"]


def run_significance_tests(
    result_root: str = "result",
    metric: str = "macro_f1",
    output_path: str | Path = "result/metrics/significance_tests.csv",
) -> list[dict[str, Any]]:
    """Compute paired tests or bootstrap CIs over available multi-seed metrics."""

    metrics = load_prediction_metrics(Path(result_root) / "predictions")
    rows: list[dict[str, Any]] = []
    for dataset in sorted({key[0] for key in metrics}):
        best_baseline = _best_baseline(metrics, dataset, metric)
        comparisons = [(a, best_baseline if b == "best_simple_baseline" else b) for a, b in COMPARISONS if b != "best_simple_baseline" or best_baseline]
        for model_a, model_b in comparisons:
            values_a = metrics.get((dataset, model_a), {})
            values_b = metrics.get((dataset, model_b), {})
            row = paired_significance_row(dataset, metric, model_a, values_a, model_b, values_b)
            rows.append(row)
    _write_csv(Path(output_path), rows)
    return rows


def load_prediction_metrics(predictions_root: str | Path) -> dict[tuple[str, str], dict[str, dict[str, float]]]:
    """Load metrics from result/predictions/{dataset}/{model}/{seed}/metrics.json."""

    output: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    for path in Path(predictions_root).glob("*/*/*/metrics.json"):
        dataset, model, seed = path.parts[-4], path.parts[-3], path.parts[-2]
        with path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        output.setdefault((dataset, model), {})[seed] = metrics
    return output


def paired_significance_row(
    dataset: str,
    metric: str,
    model_a: str,
    values_a: dict[str, dict[str, float]],
    model_b: str,
    values_b: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Build one significance row from two seed->metrics maps."""

    paired_seeds = sorted(set(values_a) & set(values_b))
    a = [_metric_value(values_a[seed], metric) for seed in paired_seeds]
    b = [_metric_value(values_b[seed], metric) for seed in paired_seeds]
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    a_vals = [x for x, _ in pairs]
    b_vals = [y for _, y in pairs]
    deltas = [x - y for x, y in pairs]
    p_value = _paired_ttest_pvalue(a_vals, b_vals) if len(pairs) >= 2 else None
    ci_low, ci_high = bootstrap_ci(deltas) if deltas else (None, None)
    return {
        "dataset": dataset,
        "metric": metric,
        "model_a": model_a,
        "model_b": model_b,
        "mean_a": mean(a_vals) if a_vals else None,
        "std_a": stdev(a_vals) if len(a_vals) > 1 else 0.0 if a_vals else None,
        "mean_b": mean(b_vals) if b_vals else None,
        "std_b": stdev(b_vals) if len(b_vals) > 1 else 0.0 if b_vals else None,
        "delta": mean(deltas) if deltas else None,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def bootstrap_ci(values: list[float], n: int = 1000, seed: int = 42, alpha: float = 0.05) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean of paired deltas."""

    if not values:
        return (None, None)  # type: ignore[return-value]
    rng = random.Random(seed)
    estimates = []
    for _ in range(n):
        sample = [rng.choice(values) for _ in values]
        estimates.append(sum(sample) / len(sample))
    estimates.sort()
    low = estimates[int((alpha / 2) * len(estimates))]
    high = estimates[int((1 - alpha / 2) * len(estimates)) - 1]
    return low, high


def _best_baseline(metrics: dict[tuple[str, str], dict[str, dict[str, float]]], dataset: str, metric: str) -> str | None:
    best_name = None
    best_score = float("-inf")
    for model in SIMPLE_BASELINES:
        values = [_metric_value(row, metric) for row in metrics.get((dataset, model), {}).values()]
        values = [value for value in values if value is not None]
        if values and mean(values) > best_score:
            best_score = mean(values)
            best_name = model
    return best_name


def _metric_value(metrics: dict[str, Any], metric: str) -> float | None:
    value = metrics.get(metric)
    if value is None and metric == "macro_f1":
        value = metrics.get("harmfulness_macro_f1")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _paired_ttest_pvalue(a: list[float], b: list[float]) -> float | None:
    try:
        from scipy import stats  # type: ignore

        return float(stats.ttest_rel(a, b, nan_policy="omit").pvalue)
    except Exception:
        return None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0]) if rows else ["dataset", "metric", "model_a", "model_b"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
