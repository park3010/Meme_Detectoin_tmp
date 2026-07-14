"""Canonical logits-only decoding for formal rhetorical tactic metrics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dataset.labels import LabelVocab
from utils.io import load_yaml


@dataclass(frozen=True)
class TacticDecodingSpec:
    """Formal decoding configuration for trainable rhetorical tactic logits."""

    schema_version: str
    label_order: list[str]
    non_none_labels: list[str]
    none_label: str
    prediction_source: str
    threshold_policy: str
    threshold_candidates: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ThresholdSelectionResult:
    """Validation-only threshold selection result."""

    selected_threshold: float
    validation_macro_f1: float | None
    validation_micro_f1: float | None
    candidate_results: list[dict[str, Any]]
    eligible_validation_samples: int
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_tactic_decoding_spec(
    config: dict[str, Any] | None = None,
    vocab_path: str | Path = "configs/label_vocab.yaml",
    label_order: list[str] | None = None,
) -> TacticDecodingSpec:
    """Resolve formal rhetorical tactic decoding spec from config/vocab."""

    cfg = config or {}
    vocab = LabelVocab.from_yaml(vocab_path)
    ignore = vocab.multi_ignore_labels.get("tactic_rhetorical", set())
    if label_order is None:
        try:
            from module.structured_interpretation_head import TacticHead
        except Exception:
            TacticHead = None  # type: ignore
        if TacticHead is not None:
            label_order = list(TacticHead.labels)
        else:
            label_order = [label for label in vocab.multi_label_fields.get("tactic_rhetorical", []) if label not in ignore]
    usable_order = [label for label in label_order if label not in ignore]
    none_label = "none"
    non_none = [label for label in usable_order if label != none_label]
    contract = (cfg.get("preflight", {}) if isinstance(cfg, dict) else {}).get("metric_contract", {})
    tactic_cfg = contract.get("tactic_rhetorical", {}) if isinstance(contract, dict) else {}
    candidates = tactic_cfg.get("threshold_candidates") or [0.5]
    return TacticDecodingSpec(
        schema_version="tactic_rhetorical_decoding_spec_v1",
        label_order=list(usable_order),
        non_none_labels=list(non_none),
        none_label=none_label,
        prediction_source="tactic_logits_sigmoid",
        threshold_policy=str(tactic_cfg.get("threshold_policy", "validation_grid_search")),
        threshold_candidates=[float(value) for value in candidates],
    )


def canonicalize_gold_tactic_labels(
    labels: Any,
    spec: TacticDecodingSpec,
    ignored_labels: set[str] | None = None,
) -> tuple[set[str], dict[str, Any]]:
    """Canonicalize gold labels into a non-none set and diagnostics."""

    ignored = set(ignored_labels or {"unknown"})
    raw_values = _as_list(labels)
    diagnostics: dict[str, Any] = {
        "raw_labels": list(raw_values),
        "ignored_labels": [],
        "unsupported_labels": [],
        "none_conflict_removed": False,
        "gold_none": False,
        "usable": True,
    }
    clean: list[str] = []
    for raw in raw_values:
        value = str(raw).strip()
        if not value:
            continue
        if value in ignored:
            diagnostics["ignored_labels"].append(value)
            continue
        if value not in spec.label_order:
            diagnostics["unsupported_labels"].append(value)
            continue
        clean.append(value)
    non_none = {label for label in clean if label != spec.none_label and label in spec.non_none_labels}
    has_none = spec.none_label in clean
    if non_none and has_none:
        diagnostics["none_conflict_removed"] = True
    if non_none:
        diagnostics["gold_none"] = False
        return non_none, diagnostics
    if has_none:
        diagnostics["gold_none"] = True
        return set(), diagnostics
    if diagnostics["ignored_labels"] or diagnostics["unsupported_labels"] or not raw_values:
        diagnostics["usable"] = False
    diagnostics["gold_none"] = False
    return set(), diagnostics


def sigmoid_probabilities(logits: list[float] | Any) -> list[float]:
    """Convert logits to numerically safe sigmoid probabilities."""

    values = _flatten_logits(logits)
    probs = []
    for value in values:
        x = float(value)
        if x >= 0:
            z = math.exp(-x)
            probs.append(1.0 / (1.0 + z))
        else:
            z = math.exp(x)
            probs.append(z / (1.0 + z))
    return probs


def decode_tactic_logits(
    logits: list[float] | Any,
    spec: TacticDecodingSpec,
    threshold: float,
) -> dict[str, Any]:
    """Decode trainable tactic logits using sigmoid and non-none thresholding."""

    values = _flatten_logits(logits)
    diagnostics: dict[str, Any] = {}
    if len(values) != len(spec.label_order):
        diagnostics["logits_length_mismatch"] = {"expected": len(spec.label_order), "actual": len(values)}
    probs = sigmoid_probabilities(values[: len(spec.label_order)])
    score_by_label = {label: probs[idx] for idx, label in enumerate(spec.label_order) if idx < len(probs)}
    selected = [label for label in spec.non_none_labels if score_by_label.get(label, 0.0) >= threshold]
    predicted_none = not selected
    return {
        "prediction_source": spec.prediction_source,
        "threshold": float(threshold),
        "probabilities": score_by_label,
        "predicted_non_none_labels": selected,
        "predicted_labels_with_none_fallback": selected if selected else [spec.none_label],
        "predicted_none": predicted_none,
        "rendered_labels_used": False,
        "diagnostics": diagnostics,
    }


def select_tactic_threshold(
    validation_logits: list[list[float]],
    validation_gold_labels: list[list[str]],
    spec: TacticDecodingSpec,
) -> ThresholdSelectionResult:
    """Select a global threshold on validation macro-F1 over non-none labels."""

    candidate_results = []
    for threshold in spec.threshold_candidates:
        metrics = compute_tactic_logits_only_metrics(validation_logits, validation_gold_labels, spec, threshold)
        candidate_results.append(
            {
                "threshold": float(threshold),
                "macro_f1": metrics.get("tactic_rhetorical_macro_f1_logits_only"),
                "micro_f1": metrics.get("tactic_rhetorical_micro_f1_logits_only"),
                "eligible_sample_count": metrics.get("tactic_rhetorical_eligible_sample_count", 0),
            }
        )

    def key(row: dict[str, Any]):
        macro = row.get("macro_f1")
        micro = row.get("micro_f1")
        threshold = float(row["threshold"])
        return (
            -1.0 if macro is None else float(macro),
            -1.0 if micro is None else float(micro),
            -abs(threshold - 0.5),
            -threshold,
        )

    best = max(candidate_results, key=key) if candidate_results else {"threshold": 0.5, "macro_f1": None, "micro_f1": None, "eligible_sample_count": 0}
    return ThresholdSelectionResult(
        selected_threshold=float(best["threshold"]),
        validation_macro_f1=best.get("macro_f1"),
        validation_micro_f1=best.get("micro_f1"),
        candidate_results=candidate_results,
        eligible_validation_samples=int(best.get("eligible_sample_count") or 0),
        diagnostics={"tie_breaking": "macro_f1,micro_f1,closest_to_0.50,lower_threshold"},
    )


def compute_tactic_logits_only_metrics(
    logits: list[list[float]],
    gold_labels: list[list[str]],
    spec: TacticDecodingSpec,
    threshold: float,
) -> dict[str, Any]:
    """Compute formal logits-only rhetorical tactic metrics."""

    pairs = []
    diagnostics = {"gold": [], "prediction": []}
    for sample_logits, sample_gold in zip(logits, gold_labels):
        gold_set, gold_diag = canonicalize_gold_tactic_labels(sample_gold, spec)
        pred = decode_tactic_logits(sample_logits, spec, threshold)
        diagnostics["gold"].append(gold_diag)
        diagnostics["prediction"].append(pred.get("diagnostics", {}))
        if not gold_diag.get("usable", True):
            continue
        pairs.append((gold_set, set(pred["predicted_non_none_labels"])))
    if not pairs:
        return _empty_formal_metrics(spec, threshold, status="no_eligible_samples", diagnostics=diagnostics)

    per_label_f1 = []
    per_label_f1_map: dict[str, float] = {}
    micro_tp = micro_fp = micro_fn = 0
    for label in spec.non_none_labels:
        tp = fp = fn = 0
        for gold_set, pred_set in pairs:
            tp += int(label in gold_set and label in pred_set)
            fp += int(label not in gold_set and label in pred_set)
            fn += int(label in gold_set and label not in pred_set)
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        label_f1 = _f1_counts(tp, fp, fn)
        per_label_f1.append(label_f1)
        per_label_f1_map[label] = label_f1
    none_tp = none_fp = none_fn = 0
    exact = 0
    for gold_set, pred_set in pairs:
        gold_none = not gold_set
        pred_none = not pred_set
        none_tp += int(gold_none and pred_none)
        none_fp += int(not gold_none and pred_none)
        none_fn += int(gold_none and not pred_none)
        exact += int(gold_set == pred_set)
    return {
        "tactic_rhetorical_formal_status": "ready",
        "tactic_rhetorical_prediction_source": spec.prediction_source,
        "tactic_rhetorical_rendered_labels_used": False,
        "tactic_rhetorical_macro_f1_logits_only": sum(per_label_f1) / len(per_label_f1) if per_label_f1 else None,
        "tactic_rhetorical_micro_f1_logits_only": _f1_counts(micro_tp, micro_fp, micro_fn),
        "tactic_rhetorical_per_label_f1_logits_only": per_label_f1_map,
        "tactic_rhetorical_none_f1": _f1_counts(none_tp, none_fp, none_fn),
        "tactic_rhetorical_exact_match_ratio": exact / len(pairs),
        "tactic_rhetorical_eligible_sample_count": len(pairs),
        "tactic_rhetorical_non_none_label_count": len(spec.non_none_labels),
        "tactic_rhetorical_validation_selected_threshold": float(threshold),
        "tactic_rhetorical_validation_macro_f1_at_selected_threshold": None,
        "tactic_rhetorical_diagnostics": _compact_diagnostics(diagnostics),
    }


def _empty_formal_metrics(spec: TacticDecodingSpec, threshold: float, status: str, diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        "tactic_rhetorical_formal_status": status,
        "tactic_rhetorical_prediction_source": spec.prediction_source,
        "tactic_rhetorical_rendered_labels_used": False,
        "tactic_rhetorical_macro_f1_logits_only": None,
        "tactic_rhetorical_micro_f1_logits_only": None,
        "tactic_rhetorical_per_label_f1_logits_only": {},
        "tactic_rhetorical_none_f1": None,
        "tactic_rhetorical_exact_match_ratio": None,
        "tactic_rhetorical_eligible_sample_count": 0,
        "tactic_rhetorical_non_none_label_count": len(spec.non_none_labels),
        "tactic_rhetorical_validation_selected_threshold": float(threshold),
        "tactic_rhetorical_validation_macro_f1_at_selected_threshold": None,
        "tactic_rhetorical_diagnostics": _compact_diagnostics(diagnostics),
    }


def _compact_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    gold = diagnostics.get("gold", [])
    prediction = diagnostics.get("prediction", [])
    return {
        "ignored_label_count": sum(len(item.get("ignored_labels", [])) for item in gold if isinstance(item, dict)),
        "unsupported_label_count": sum(len(item.get("unsupported_labels", [])) for item in gold if isinstance(item, dict)),
        "none_conflict_count": sum(1 for item in gold if isinstance(item, dict) and item.get("none_conflict_removed")),
        "logits_length_mismatch_count": sum(1 for item in prediction if isinstance(item, dict) and item.get("logits_length_mismatch")),
    }


def _f1_counts(tp: int, fp: int, fn: int) -> float:
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _as_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _flatten_logits(logits: list[float] | Any) -> list[float]:
    if logits is None:
        return []
    if isinstance(logits, dict):
        if isinstance(logits.get("values"), list):
            return [float(value) for value in logits["values"]]
        if isinstance(logits.get("preview"), list) and "shape" not in logits:
            return [float(value) for value in logits["preview"]]
        return []
    try:
        import torch

        if isinstance(logits, torch.Tensor):
            return [float(value) for value in logits.detach().cpu().flatten().tolist()]
    except Exception:
        pass
    if isinstance(logits, (list, tuple)):
        return [float(value) for value in logits]
    return []


def extract_tactic_logits(record: dict[str, Any]) -> list[float] | None:
    """Extract full trainable tactic logits from a prediction record."""

    candidates = [
        record.get("tactic_rhetorical_logits"),
        (record.get("training_hooks") or {}).get("tactic_logits"),
        (record.get("tactic") or {}).get("logits"),
    ]
    for candidate in candidates:
        values = _flatten_logits(candidate)
        if values:
            return values
    return None


def extract_gold_tactic_labels(record: dict[str, Any]) -> list[str]:
    """Extract normalized gold rhetorical tactic labels from a prediction record."""

    return [str(item) for item in _as_list((record.get("gold_tactic") or {}).get("tactic_rhetorical"))]


def extract_tactic_label_order(record: dict[str, Any]) -> list[str]:
    """Extract serialized tactic-logit label order from a prediction record."""

    candidates = [
        record.get("tactic_rhetorical_label_order"),
        ((record.get("output_provenance") or {}).get("label_spaces") or {}).get("tactic_rhetorical"),
        ((record.get("output_provenance") or {}).get("label_spaces") or {}).get("tactic"),
        ((record.get("training_hooks") or {}).get("label_spaces") or {}).get("tactic_rhetorical"),
    ]
    for candidate in candidates:
        values = [str(item) for item in _as_list(candidate) if str(item).strip()]
        if values:
            return values
    return []


__all__ = [
    "TacticDecodingSpec",
    "ThresholdSelectionResult",
    "resolve_tactic_decoding_spec",
    "canonicalize_gold_tactic_labels",
    "sigmoid_probabilities",
    "decode_tactic_logits",
    "select_tactic_threshold",
    "compute_tactic_logits_only_metrics",
    "extract_tactic_logits",
    "extract_gold_tactic_labels",
    "extract_tactic_label_order",
]
