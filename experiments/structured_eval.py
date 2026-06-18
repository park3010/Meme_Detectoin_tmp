"""Structured interpretation evaluation for framework predictions."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from experiments.metrics import compute_harmfulness_metrics
from experiments.prediction_io import load_prediction_records
from experiments.progress import progress_iter
from utils.annotation_utils import parse_bool
from utils.io import write_json
from utils.text_utils import jaccard_similarity


def evaluate_structured_predictions(
    records: list[dict[str, Any]],
    evidence_k: int = 3,
    disable_tqdm: bool = False,
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
    metrics["tactic_multimodal_relation_macro_f1"] = macro_f1(
        [record.get("gold_tactic", {}).get("tactic_multimodal_relation") for record in records],
        [record.get("tactic", {}).get("multimodal_relation") for record in records],
    )
    evidence_scores = weak_evidence_scores(records, k=evidence_k, disable_tqdm=disable_tqdm)
    metrics.update(evidence_scores)
    return metrics


def evaluate_prediction_file(
    prediction_path: str | Path,
    dataset: str,
    output_root: str | Path = "result/metrics",
    disable_tqdm: bool = False,
) -> dict[str, Any]:
    """Evaluate one prediction file and save per-dataset JSON."""

    records = load_prediction_records(prediction_path)
    metrics = evaluate_structured_predictions(records, disable_tqdm=disable_tqdm)
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
) -> list[dict[str, Any]]:
    """Scan prediction files and evaluate structured outputs for each run."""

    rows: list[dict[str, Any]] = []
    root = Path(predictions_root)
    for path in sorted(root.glob("*/*/*/final_predictions.jsonl")):
        dataset, model, seed = path.parts[-4], path.parts[-3], path.parts[-2]
        metrics = evaluate_structured_predictions(load_prediction_records(path), disable_tqdm=disable_tqdm)
        rows.append({"dataset": dataset, "model": model, "seed": seed, **metrics})
    return rows


def write_structured_aggregate_tables(
    predictions_root: str | Path = "result/predictions",
    output_root: str | Path = "result/metrics",
    disable_tqdm: bool = False,
) -> tuple[Path, Path]:
    """Write per-run and mean/std structured interpretation metric tables."""

    rows = collect_structured_metric_rows(predictions_root, disable_tqdm=disable_tqdm)
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
) -> dict[str, float | None]:
    hits = 0
    precision_sum = 0.0
    total = 0
    for record in progress_iter(records, desc="structured evidence", leave=False, disable=disable_tqdm):
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
