"""Automatic and human-template rationale quality evaluation."""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from dataset import MemeDataset
from experiments.evaluation import compute_harmfulness_metrics
from experiments.prediction_io import load_prediction_records
from experiments.progress import ProgressConfig, progress_iter
from experiments.splits import label_to_int
from module.losses import extract_supervision_from_annotation
from module.runner import HarmfulMemePipeline
from utils.io import load_yaml, write_jsonl
from utils.text_utils import jaccard_similarity, normalize_text


GENERIC_TOKENS = {
    "meme",
    "image",
    "text",
    "content",
    "suggests",
    "indicates",
    "because",
    "target",
    "intent",
    "tactic",
    "harmful",
    "non",
    "evidence",
}


def run_rationale_evaluation(
    dataset: str = "all",
    model: str = "ours_full",
    seed: int = 42,
    result_root: str = "result",
    disable_tqdm: bool = False,
    progress: ProgressConfig | None = None,
) -> list[dict[str, Any]]:
    """Evaluate rationale quality proxies and export a human-rating template."""

    records = _load_records(dataset, model, seed, result_root)
    progress_config = progress or ProgressConfig(disable=True if disable_tqdm else None)
    rows = [
        rationale_metrics(record) | {"dataset": record.get("dataset_name"), "model": model, "seed": seed}
        for record in progress_iter(
            records,
            desc="rationale eval",
            config=progress_config,
            position=2,
            leave=progress_config.leave_batch,
        )
    ]
    metrics_path = Path(result_root) / "metrics" / "rationale_eval.csv"
    _write_csv(metrics_path, rows)
    _write_human_template(Path(result_root) / "analysis" / "rationale_human_eval_template.csv", records)
    return rows


def rationale_metrics(record: dict[str, Any]) -> dict[str, Any]:
    """Compute automatic proxy metrics for one rationale."""

    rationale = str(record.get("rationale", ""))
    evidence_texts = _selected_evidence_texts(record)
    context = " ".join([str(record.get("ocr_text_full", "")), *evidence_texts])
    entities = _entities(rationale)
    allowed_entities = set(_entities(context))
    hallucinated = [entity for entity in entities if entity not in allowed_entities]
    return {
        "sample_id": record.get("sample_id"),
        "evidence_mention_rate": _evidence_mention_rate(rationale, evidence_texts),
        "selected_evidence_coverage": _selected_evidence_coverage(rationale, evidence_texts),
        "label_contradiction": _label_contradiction(record, rationale),
        "hallucination_proxy": len(hallucinated) / max(1, len(entities)),
        "hallucinated_entities": "; ".join(hallucinated),
        "specificity_score": _specificity(rationale),
        "label_consistency": _label_consistency(record, rationale),
    }


def _load_records(dataset: str, model: str, seed: int, result_root: str) -> list[dict[str, Any]]:
    root = Path(result_root) / "predictions"
    paths = sorted(root.glob(f"*/{model}/{seed}/final_predictions.jsonl")) if dataset == "all" else [root / dataset / model / str(seed) / "final_predictions.jsonl"]
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(load_prediction_records(path))
    return records


def _selected_evidence_texts(record: dict[str, Any]) -> list[str]:
    evidence: list[str] = []
    for group in (record.get("supporting_evidence", {}) or {}).values():
        evidence.extend(str(item.get("text", "")) for item in group if str(item.get("text", "")).strip())
    return evidence


def _evidence_mention_rate(rationale: str, evidence_texts: list[str]) -> float | None:
    if not evidence_texts:
        return None
    return sum(1 for text in evidence_texts if _overlaps(rationale, text)) / len(evidence_texts)


def _selected_evidence_coverage(rationale: str, evidence_texts: list[str]) -> float | None:
    if not evidence_texts:
        return None
    tokens = set(normalize_text(rationale).split())
    evidence_tokens = set(token for text in evidence_texts for token in normalize_text(text).split())
    if not evidence_tokens:
        return None
    return len(tokens & evidence_tokens) / len(evidence_tokens)


def _label_contradiction(record: dict[str, Any], rationale: str) -> int:
    lowered = rationale.lower()
    harmful = str(record.get("pred_harmfulness", "")).lower()
    if harmful == "harmful" and any(term in lowered for term in ["not harmful", "benign", "harmless"]):
        return 1
    if harmful in {"non_harmful", "non-harmful"} and any(term in lowered for term in ["attacks", "dehumanizes", "threatens"]):
        return 1
    return 0


def _specificity(rationale: str) -> float:
    tokens = [token for token in normalize_text(rationale).split() if token]
    if not tokens:
        return 0.0
    specific = [token for token in tokens if token not in GENERIC_TOKENS and len(token) > 3]
    return len(specific) / len(tokens)


def _label_consistency(record: dict[str, Any], rationale: str) -> float:
    lowered = rationale.lower()
    labels = [
        record.get("pred_harmfulness"),
        (record.get("target", {}) or {}).get("label"),
        (record.get("intent", {}) or {}).get("primary"),
        (record.get("tactic", {}) or {}).get("multimodal_relation"),
    ]
    clean = [str(label).replace("_", " ").lower() for label in labels if label]
    if not clean:
        return 0.0
    return sum(1 for label in clean if label in lowered or any(part in lowered for part in label.split())) / len(clean)


def _overlaps(a: str, b: str) -> bool:
    return jaccard_similarity(a, b) >= 0.15 or normalize_text(b) in normalize_text(a)


def _entities(text: str) -> list[str]:
    return sorted(set(re.findall(r"\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*\b", text)))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row}) if rows else ["sample_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_human_template(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "sample_id",
        "image_path",
        "OCR text",
        "predicted harmfulness",
        "predicted target",
        "predicted intent",
        "predicted tactic",
        "selected evidence",
        "rationale",
        "faithfulness_score",
        "usefulness_score",
        "specificity_score",
        "hallucination_flag",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sample_id": record.get("sample_id"),
                    "image_path": record.get("image_path"),
                    "OCR text": record.get("ocr_text_full", ""),
                    "predicted harmfulness": record.get("pred_harmfulness"),
                    "predicted target": record.get("target"),
                    "predicted intent": record.get("intent"),
                    "predicted tactic": record.get("tactic"),
                    "selected evidence": " | ".join(_selected_evidence_texts(record)),
                    "rationale": record.get("rationale", ""),
                    "faithfulness_score": "",
                    "usefulness_score": "",
                    "specificity_score": "",
                    "hallucination_flag": "",
                    "notes": "",
                }
            )


# =============================================================================
# Verifier candidate-quality evaluation
# =============================================================================

def run_verifier_evaluation(
    dataset_name: str = "harm_c",
    seed: int = 42,
    config_path: str = "configs/config.yaml",
    output_root: str = "result",
    limit: int | None = None,
    disable_tqdm: bool = False,
    progress: ProgressConfig | None = None,
) -> dict[str, Any]:
    """Run Stage A-B-C and evaluate verifier decisions with weak labels."""

    _ = seed
    cfg = load_yaml(config_path)
    dataset = MemeDataset(
        dataset_root=cfg.get("paths", {}).get("dataset_root", "dataset/source"),
        annotation_root=cfg.get("paths", {}).get("annotation_root", "dataset/annotation"),
        dataset_names=[dataset_name],
        keep_missing_images=True,
        limit=limit,
    )
    pipeline = HarmfulMemePipeline(cfg).eval()
    examples: list[dict[str, Any]] = []
    progress_config = progress or ProgressConfig(disable=True if disable_tqdm else None)
    with torch.no_grad():
        for sample in progress_iter(
            dataset,
            desc=f"verifier {dataset_name}",
            config=progress_config,
            position=2,
            leave=progress_config.leave_batch,
            total=len(dataset),
        ):
            if label_to_int(sample.get("raw_label")) is None:
                continue
            stage_a = pipeline.stage_a(sample)
            stage_b = pipeline.stage_b(stage_a, sample.get("ocr_text_full", ""))
            stage_c = pipeline.stage_c(stage_a, stage_b)
            examples.extend(_candidate_examples(sample, stage_b, stage_c))
    metrics = verifier_metrics(examples)
    _write_metric_row(Path(output_root) / "metrics" / "verifier_eval.csv", dataset_name, seed, metrics)
    write_jsonl(Path(output_root) / "analysis" / "verifier_examples.jsonl", examples, compact_tensors=True)
    return metrics


def verifier_metrics(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute relevance, support, credibility, and weak compatibility metrics."""

    if not examples:
        return {}
    relevance = compute_harmfulness_metrics(
        [int(row.get("weak_relevance_label", 0)) for row in examples],
        [int(row.get("final_decision") == "accepted") for row in examples],
        [float(row.get("relevance_score", 0.0)) for row in examples],
    )
    support_gold = [str(row.get("weak_support_label", "insufficient")) for row in examples]
    support_pred = [str(row.get("support_label", "insufficient")) for row in examples]
    support = _macro_f1_and_confusion(support_gold, support_pred, ["support", "contradict", "insufficient"])
    accepted = [row for row in examples if row.get("final_decision") == "accepted"]
    accepted_precision = (
        sum(1 for row in accepted if row.get("weak_relevance_label")) / len(accepted)
        if accepted
        else None
    )
    rejection_rate = sum(1 for row in examples if row.get("final_decision") == "rejected") / len(examples)
    temporal_rows = [row for row in examples if row.get("temporal_label") is not None]
    temporal_error = (
        sum(1 for row in temporal_rows if row.get("temporal_label") != row.get("temporal_prediction")) / len(temporal_rows)
        if temporal_rows
        else None
    )
    compatibility_scores = [float(row.get("cultural_compatibility_score", 0.0)) for row in examples if row.get("cultural_compatibility_score") is not None]
    return {
        "relevance_precision": relevance.get("precision"),
        "relevance_recall": relevance.get("recall"),
        "relevance_f1": relevance.get("macro_f1"),
        "relevance_roc_auc": relevance.get("roc_auc"),
        "support_macro_f1": support["macro_f1"],
        "support_confusion": support["confusion"],
        "accepted_precision": accepted_precision,
        "rejection_rate": rejection_rate,
        "temporal_error_rate": temporal_error,
        "cultural_language_compatibility": sum(compatibility_scores) / len(compatibility_scores) if compatibility_scores else None,
    }


def _candidate_examples(sample: dict[str, Any], stage_b: Any, stage_c: Any) -> list[dict[str, Any]]:
    supervision = extract_supervision_from_annotation(sample)
    gold_texts = [normalize_text(text) for text in supervision.get("evidence_text", []) if normalize_text(text)]
    verified = {item.knowledge_id: item for item in stage_c.verified_items}
    rows: list[dict[str, Any]] = []
    for candidate in stage_b.knowledge_candidates:
        item = verified.get(candidate.candidate_id)
        score = float(item.relevance_score if item else candidate.score)
        support_label = str(item.support_label if item else "insufficient")
        validity = float(item.validity_score if item else 0.0)
        weak_relevant = _weak_relevance(candidate.text, gold_texts, supervision)
        rows.append(
            {
                "sample_id": sample.get("sample_id"),
                "dataset_name": sample.get("dataset_name"),
                "candidate_id": candidate.candidate_id,
                "candidate_text": candidate.text,
                "candidate_source": candidate.source,
                "candidate_type": candidate.candidate_type,
                "query": candidate.query,
                "relevance_score": score,
                "support_label": support_label,
                "validity_score": validity,
                "final_score": float(item.final_score if item else 0.0),
                "final_decision": "accepted" if item else "rejected",
                "weak_relevance_label": weak_relevant,
                "weak_support_label": _weak_support_label(candidate.text, weak_relevant),
                "temporal_label": None,
                "temporal_prediction": None,
                "cultural_compatibility_score": _weak_cultural_score(candidate.text, sample.get("ocr_text_full", "")),
                "reason": (item.metadata if item else candidate.metadata),
            }
        )
    return rows


def _weak_relevance(text: str, gold_texts: list[str], supervision: dict[str, Any]) -> int:
    clean = normalize_text(text)
    if gold_texts and any(jaccard_similarity(clean, gold) >= 0.18 or gold in clean for gold in gold_texts):
        return 1
    labels = [
        supervision.get("target_granularity"),
        supervision.get("intent_primary"),
        supervision.get("tactic_multimodal_relation"),
    ]
    return int(any(str(label).lower().replace("_", " ") in clean for label in labels if label))


def _weak_support_label(text: str, relevant: int) -> str:
    lowered = text.lower()
    if any(term in lowered for term in ["not ", "false", "debunk", "contradict", "myth"]):
        return "contradict"
    return "support" if relevant else "insufficient"


def _weak_cultural_score(candidate_text: str, ocr_text: str) -> float:
    candidate = normalize_text(candidate_text)
    ocr = normalize_text(ocr_text)
    if not candidate or not ocr:
        return 0.5
    return 1.0 if any(token in candidate for token in ocr.split()[:20]) else 0.5


def _macro_f1_and_confusion(gold: list[str], pred: list[str], labels: list[str]) -> dict[str, Any]:
    confusion: dict[str, dict[str, int]] = {label: {other: 0 for other in labels} for label in labels}
    f1s = []
    for g, p in zip(gold, pred):
        if g in confusion and p in confusion[g]:
            confusion[g][p] += 1
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1s.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return {"macro_f1": sum(f1s) / len(f1s), "confusion": confusion}


def _write_metric_row(path: Path, dataset: str, seed: int, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"dataset": dataset, "seed": seed, **{k: v for k, v in metrics.items() if k != "support_confusion"}}
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
