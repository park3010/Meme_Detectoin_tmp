"""Automatic and human-template rationale quality evaluation."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from experiments.prediction_io import load_prediction_records
from experiments.progress import ProgressConfig, progress_iter
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
