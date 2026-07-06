"""Evaluation utilities for harmfulness and structured interpretation."""

from __future__ import annotations

import csv
from math import isnan
from pathlib import Path
from typing import Any, Iterable

from experiments.prediction_io import load_prediction_records
from experiments.progress import ProgressConfig, progress_iter
from experiments.tactic_decoding import (
    TacticDecodingSpec,
    compute_tactic_logits_only_metrics,
    decode_tactic_logits,
    extract_gold_tactic_labels,
    extract_tactic_label_order,
    extract_tactic_logits,
    resolve_tactic_decoding_spec,
    select_tactic_threshold,
)
from utils.annotation_utils import parse_bool
from utils.io import read_jsonl, write_json
from utils.text_utils import jaccard_similarity


# =============================================================================
# Harmfulness metrics
# =============================================================================

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


# =============================================================================
# Structured interpretation evaluation
# =============================================================================

def evaluate_structured_predictions(
    records: list[dict[str, Any]],
    evidence_k: int = 3,
    disable_tqdm: bool = False,
    progress: ProgressConfig | None = None,
) -> dict[str, Any]:
    """Evaluate harmfulness, target, intent, tactic, and weak evidence matching."""

    if not records:
        return {}
    metrics: dict[str, Any] = {}
    labeled = [record for record in records if record.get("gold_label") is not None]
    metrics.update(
        {
            f"harmfulness_{key}": value
            for key, value in compute_harmfulness_metrics(
                [record["gold_label"] for record in labeled],
                [record["pred_label"] for record in labeled],
                [record.get("prob_harmful", 0.0) for record in labeled],
            ).items()
        }
    )

    metrics["target_presence_macro_f1"] = macro_f1(
        [record.get("gold_target", {}).get("target_presence") for record in records],
        [record.get("target", {}).get("presence") for record in records],
    )
    metrics["target_granularity_macro_f1"] = macro_f1(
        [record.get("gold_target", {}).get("target_granularity") for record in records],
        [record.get("target", {}).get("granularity") for record in records],
    )
    attr_scores = multilabel_f1(
        [as_list(record.get("gold_target", {}).get("protected_attribute")) for record in records],
        [as_list(record.get("target", {}).get("attributes")) for record in records],
    )
    metrics["protected_attribute_micro_f1"] = attr_scores["micro_f1"]
    metrics["protected_attribute_macro_f1"] = attr_scores["macro_f1"]

    metrics["intent_primary_macro_f1"] = macro_f1(
        [record.get("gold_intent", {}).get("intent_primary") for record in records],
        [record.get("intent", {}).get("primary") for record in records],
    )
    metrics["stance_accuracy"] = accuracy(
        [record.get("gold_intent", {}).get("stance") for record in records],
        [record.get("intent", {}).get("stance") for record in records],
    )
    metrics["stance_macro_f1"] = macro_f1(
        [record.get("gold_intent", {}).get("stance") for record in records],
        [record.get("intent", {}).get("stance") for record in records],
    )
    metrics["secondary_intent_f1"] = macro_f1(
        [record.get("gold_intent", {}).get("secondary_intent") for record in records],
        [_first(record.get("intent", {}).get("secondary")) for record in records],
    )
    metrics["background_knowledge_needed_accuracy"] = accuracy(
        [_bool_label(record.get("gold_intent", {}).get("background_knowledge_needed")) for record in records],
        [_bool_label(record.get("intent", {}).get("background_knowledge_needed")) for record in records],
    )
    metrics["background_knowledge_needed_f1"] = macro_f1(
        [_bool_label(record.get("gold_intent", {}).get("background_knowledge_needed")) for record in records],
        [_bool_label(record.get("intent", {}).get("background_knowledge_needed")) for record in records],
    )

    tactic_scores = multilabel_f1(
        [as_list(record.get("gold_tactic", {}).get("tactic_rhetorical")) for record in records],
        [as_list(record.get("tactic", {}).get("rhetorical")) for record in records],
    )
    metrics["tactic_rhetorical_micro_f1"] = tactic_scores["micro_f1"]
    metrics["tactic_rhetorical_macro_f1"] = tactic_scores["macro_f1"]
    metrics["tactic_rhetorical_legacy_prediction_source"] = "rendered_or_heuristic_legacy"
    metrics["tactic_rhetorical_rendered_or_heuristic_legacy_micro_f1"] = tactic_scores["micro_f1"]
    metrics["tactic_rhetorical_rendered_or_heuristic_legacy_macro_f1"] = tactic_scores["macro_f1"]
    metrics.update(evaluate_tactic_rhetorical_logits_only(records))
    metrics["tactic_multimodal_relation_macro_f1"] = macro_f1(
        [record.get("gold_tactic", {}).get("tactic_multimodal_relation") for record in records],
        [record.get("tactic", {}).get("multimodal_relation") for record in records],
    )
    evidence_scores = weak_evidence_scores(records, k=evidence_k, disable_tqdm=disable_tqdm, progress=progress)
    metrics.update(evidence_scores)
    return metrics


def evaluate_tactic_rhetorical_logits_only(
    records: list[dict[str, Any]],
    *,
    threshold: float | None = None,
    threshold_selection_records: list[dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
    vocab_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate formal rhetorical tactic metrics from trainable logits only.

    This path intentionally ignores rendered fields such as ``tactic.rhetorical``
    and heuristic cue lists. If no validation-selected or explicit threshold is
    available, it reports a blocked status instead of falling back to rendering.
    """

    if not records:
        spec = resolve_tactic_decoding_spec(config, vocab_path=vocab_path or "configs/label_vocab.yaml")
        return _blocked_tactic_metrics(spec, reason="empty_records")

    label_order = _first_tactic_label_order(records)
    if not label_order and threshold_selection_records:
        label_order = _first_tactic_label_order(threshold_selection_records)
    spec = resolve_tactic_decoding_spec(
        config,
        vocab_path=vocab_path or "configs/label_vocab.yaml",
        label_order=label_order or None,
    )
    logits, gold, missing_count = _formal_tactic_inputs(records)
    if not logits:
        blocked = _blocked_tactic_metrics(spec, reason="missing_tactic_logits")
        blocked["tactic_rhetorical_missing_logits_count"] = missing_count
        return blocked

    selection = None
    threshold_source = "fixed_override"
    if threshold is None:
        if threshold_selection_records is None:
            blocked = _blocked_tactic_metrics(spec, reason="missing_validation_threshold")
            blocked["tactic_rhetorical_available_logit_record_count"] = len(logits)
            return blocked
        validation_logits, validation_gold, validation_missing = _formal_tactic_inputs(threshold_selection_records)
        if not validation_logits:
            blocked = _blocked_tactic_metrics(spec, reason="missing_validation_logits")
            blocked["tactic_rhetorical_missing_validation_logits_count"] = validation_missing
            return blocked
        selection = select_tactic_threshold(validation_logits, validation_gold, spec)
        threshold = selection.selected_threshold
        threshold_source = "validation_grid_search"

    metrics = compute_tactic_logits_only_metrics(logits, gold, spec, float(threshold))
    metrics["tactic_rhetorical_threshold_source"] = threshold_source
    metrics["tactic_rhetorical_missing_logits_count"] = missing_count
    if selection is not None:
        metrics["tactic_rhetorical_validation_macro_f1_at_selected_threshold"] = selection.validation_macro_f1
        metrics["tactic_rhetorical_validation_micro_f1_at_selected_threshold"] = selection.validation_micro_f1
        metrics["tactic_rhetorical_validation_eligible_sample_count"] = selection.eligible_validation_samples
    return metrics


def attach_formal_tactic_traces(
    records: list[dict[str, Any]],
    spec: TacticDecodingSpec,
    threshold: float,
) -> list[dict[str, Any]]:
    """Attach evaluation-only formal tactic traces to prediction records."""

    traced: list[dict[str, Any]] = []
    for record in records:
        updated = dict(record)
        evaluation = dict(updated.get("evaluation") or {})
        logits = extract_tactic_logits(updated)
        if logits is None:
            trace = {
                "prediction_source": spec.prediction_source,
                "threshold": float(threshold),
                "predicted_non_none_labels": [],
                "predicted_labels_with_none_fallback": [],
                "predicted_none": None,
                "rendered_labels_used": False,
                "status": "blocked",
                "blocked_reason": "missing_tactic_logits",
            }
        else:
            trace = decode_tactic_logits(logits, spec, threshold)
            trace["status"] = "ready"
        evaluation["tactic_rhetorical_formal"] = trace
        updated["evaluation"] = evaluation
        traced.append(updated)
    return traced


def _formal_tactic_inputs(records: list[dict[str, Any]]) -> tuple[list[list[float]], list[list[str]], int]:
    logits: list[list[float]] = []
    gold: list[list[str]] = []
    missing_count = 0
    for record in records:
        current = extract_tactic_logits(record)
        if current is None:
            missing_count += 1
            continue
        logits.append(current)
        gold.append(extract_gold_tactic_labels(record))
    return logits, gold, missing_count


def _first_tactic_label_order(records: list[dict[str, Any]]) -> list[str]:
    for record in records:
        values = extract_tactic_label_order(record)
        if values:
            return values
    return []


def _blocked_tactic_metrics(spec: TacticDecodingSpec, reason: str) -> dict[str, Any]:
    return {
        "tactic_rhetorical_formal_status": "blocked",
        "tactic_rhetorical_prediction_source": spec.prediction_source,
        "tactic_rhetorical_rendered_labels_used": False,
        "tactic_rhetorical_blocked_reason": reason,
        "tactic_rhetorical_macro_f1_logits_only": None,
        "tactic_rhetorical_micro_f1_logits_only": None,
        "tactic_rhetorical_none_f1": None,
        "tactic_rhetorical_exact_match_ratio": None,
        "tactic_rhetorical_eligible_sample_count": 0,
        "tactic_rhetorical_non_none_label_count": len(spec.non_none_labels),
        "tactic_rhetorical_validation_selected_threshold": None,
        "tactic_rhetorical_validation_macro_f1_at_selected_threshold": None,
    }


def evaluate_prediction_file(
    prediction_path: str | Path,
    dataset: str,
    output_root: str | Path = "result/metrics",
    disable_tqdm: bool = False,
    progress: ProgressConfig | None = None,
) -> dict[str, Any]:
    """Evaluate one prediction file and save per-dataset JSON."""

    records = load_prediction_records(prediction_path)
    metrics = evaluate_structured_predictions(records, disable_tqdm=disable_tqdm, progress=progress)
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / f"structured_interpretation_{dataset}.json", metrics)
    return metrics


def write_structured_csv(rows: list[dict[str, Any]], output_path: str | Path = "result/metrics/structured_interpretation.csv") -> Path:
    """Write structured metrics rows to CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def collect_structured_metric_rows(
    predictions_root: str | Path = "result/predictions",
    disable_tqdm: bool = False,
    progress: ProgressConfig | None = None,
) -> list[dict[str, Any]]:
    """Scan prediction files and evaluate structured outputs for each run."""

    rows: list[dict[str, Any]] = []
    root = Path(predictions_root)
    paths = sorted(root.glob("*/*/*/final_predictions.jsonl"))
    progress_config = progress or ProgressConfig(disable=True if disable_tqdm else None)
    for path in progress_iter(paths, desc="structured runs", config=progress_config, leave=progress_config.leave_batch):
        dataset, model, seed = path.parts[-4], path.parts[-3], path.parts[-2]
        metrics = evaluate_structured_predictions(load_prediction_records(path), disable_tqdm=disable_tqdm, progress=progress_config)
        rows.append({"dataset": dataset, "model": model, "seed": seed, **metrics})
    return rows


def write_structured_aggregate_tables(
    predictions_root: str | Path = "result/predictions",
    output_root: str | Path = "result/metrics",
    disable_tqdm: bool = False,
    progress: ProgressConfig | None = None,
) -> tuple[Path, Path]:
    """Write per-run and mean/std structured interpretation metric tables."""

    rows = collect_structured_metric_rows(predictions_root, disable_tqdm=disable_tqdm, progress=progress)
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_run = write_structured_csv(rows, output_dir / "structured_interpretation.csv")
    mean_std = output_dir / "structured_interpretation_mean_std.csv"
    _write_mean_std(rows, mean_std)
    return per_run, mean_std


def macro_f1(gold: list[Any], pred: list[Any]) -> float | None:
    pairs = [(str(g), str(p)) for g, p in zip(gold, pred) if not _is_missing_label(g)]
    if not pairs:
        return None
    labels = sorted({g for g, _ in pairs} | {p for _, p in pairs})
    f1s = []
    for label in labels:
        tp = sum(1 for g, p in pairs if g == label and p == label)
        fp = sum(1 for g, p in pairs if g != label and p == label)
        fn = sum(1 for g, p in pairs if g == label and p != label)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1s.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return sum(f1s) / len(f1s)


def accuracy(gold: list[Any], pred: list[Any]) -> float | None:
    pairs = [(g, p) for g, p in zip(gold, pred) if not _is_missing_label(g)]
    if not pairs:
        return None
    return sum(1 for g, p in pairs if str(g) == str(p)) / len(pairs)


def multilabel_f1(gold: list[list[Any]], pred: list[list[Any]]) -> dict[str, float | None]:
    labels = sorted({str(item) for row in gold + pred for item in row if item not in {None, "", "none"}})
    if not labels:
        return {"micro_f1": None, "macro_f1": None}
    tp = fp = fn = 0
    macro = []
    for label in labels:
        label_tp = label_fp = label_fn = 0
        for gold_items, pred_items in zip(gold, pred):
            gset = {str(item) for item in gold_items}
            pset = {str(item) for item in pred_items}
            label_tp += int(label in gset and label in pset)
            label_fp += int(label not in gset and label in pset)
            label_fn += int(label in gset and label not in pset)
        tp += label_tp
        fp += label_fp
        fn += label_fn
        precision = label_tp / max(1, label_tp + label_fp)
        recall = label_tp / max(1, label_tp + label_fn)
        macro.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    micro_precision = tp / max(1, tp + fp)
    micro_recall = tp / max(1, tp + fn)
    micro = 0.0 if micro_precision + micro_recall == 0 else 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
    return {"micro_f1": micro, "macro_f1": sum(macro) / len(macro)}


def weak_evidence_scores(
    records: list[dict[str, Any]],
    k: int = 3,
    disable_tqdm: bool = False,
    progress: ProgressConfig | None = None,
) -> dict[str, float | None]:
    hits = 0
    precision_sum = 0.0
    total = 0
    progress_config = progress or ProgressConfig(disable=True if disable_tqdm else None)
    for record in progress_iter(
        records,
        desc="structured evidence",
        config=progress_config,
        position=2,
        leave=progress_config.leave_batch,
    ):
        gold_texts = [text for text in as_list(record.get("gold_evidence_text")) if str(text).strip()]
        if not gold_texts:
            continue
        selected = []
        for group in record.get("supporting_evidence", {}).values():
            selected.extend([str(item.get("text", "")) for item in group[:k]])
        selected = selected[:k]
        if not selected:
            total += 1
            continue
        matched = 0
        for text in selected:
            if any(jaccard_similarity(text, gold) >= 0.2 or str(gold).lower() in text.lower() for gold in gold_texts):
                matched += 1
        hits += int(matched > 0)
        precision_sum += matched / max(1, len(selected))
        total += 1
    if total == 0:
        return {"evidence_hit_at_k": None, "evidence_precision_at_k": None}
    return {"evidence_hit_at_k": hits / total, "evidence_precision_at_k": precision_sum / total}


def as_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _first(value: Any) -> Any:
    values = as_list(value)
    return values[0] if values else None


def _bool_label(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return "true" if parse_bool(value, default=False) else "false"


def _is_missing_label(value: Any) -> bool:
    if value is None or value == "":
        return True
    if isinstance(value, (list, tuple, set)):
        return not value
    return str(value).strip().lower() in {"unknown", "null", "n/a", "na"}


def _write_mean_std(rows: list[dict[str, Any]], path: Path) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row.get("dataset")), str(row.get("model"))), []).append(row)
    metric_names = sorted({key for row in rows for key, value in row.items() if key not in {"dataset", "model", "seed"} and isinstance(value, (int, float))})
    columns = ["dataset", "model", "num_seeds", *[f"{name}_{stat}" for name in metric_names for stat in ("mean", "std")]]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for (dataset, model), group in sorted(groups.items()):
            row: dict[str, Any] = {"dataset": dataset, "model": model, "num_seeds": len(group)}
            for name in metric_names:
                values = [float(item[name]) for item in group if isinstance(item.get(name), (int, float))]
                mean = sum(values) / len(values) if values else None
                std = _std(values) if len(values) > 1 else 0.0 if values else None
                row[f"{name}_mean"] = mean
                row[f"{name}_std"] = std
            writer.writerow(row)


def _std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5


# =============================================================================
# Prediction-file evaluation
# =============================================================================


def evaluate_harmfulness_prediction_file(path: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    """Evaluate one harmfulness prediction JSONL file and optionally save metrics JSON."""

    records = read_jsonl(path)
    metrics = compute_harmfulness_metrics(
        [int(record["gold_label"]) for record in records],
        [int(record["pred_label"]) for record in records],
        [float(record["prob_harmful"]) for record in records],
    )
    if output_path is not None:
        write_json(output_path, metrics)
    return metrics


__all__ = [
    "compute_harmfulness_metrics",
    "metrics_to_csv_row",
    "summarize_mean_std",
    "evaluate_structured_predictions",
    "evaluate_tactic_rhetorical_logits_only",
    "attach_formal_tactic_traces",
    "evaluate_prediction_file",
    "evaluate_harmfulness_prediction_file",
    "write_structured_aggregate_tables",
    "write_structured_csv",
    "collect_structured_metric_rows",
    "macro_f1",
    "accuracy",
    "multilabel_f1",
]
