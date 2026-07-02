"""Metric-contract helpers for Experiment 0 preflight."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dataset.labels import LabelVocab
from utils.io import load_yaml, write_json


def resolve_metric_contract(config: dict[str, Any], vocab_path: str | Path = "configs/label_vocab.yaml") -> dict[str, Any]:
    """Resolve the canonical structured-metric contract from config and vocab."""

    preflight_cfg = config.get("preflight", {}) if isinstance(config, dict) else {}
    contract_cfg = dict(preflight_cfg.get("metric_contract", {}) or {})
    vocab = LabelVocab.from_yaml(vocab_path)
    contract: dict[str, Any] = {
        "schema_version": contract_cfg.get("schema_version", "structured_metric_contract_v1"),
        "fields": {},
        "implementation_status": "ready",
        "missing_capabilities": [],
    }
    for field, spec in contract_cfg.items():
        if field == "schema_version" or not isinstance(spec, dict):
            continue
        field_contract = dict(spec)
        field_contract["labels"] = _labels_for_field(vocab, field)
        contract["fields"][field] = field_contract

    tactic = contract["fields"].get("tactic_rhetorical")
    if tactic:
        tactic["formal_prediction_source"] = "trainable_tactic_logits_sigmoid"
        tactic["not_allowed_prediction_sources"] = [
            "top1_logits_plus_heuristic_cues",
            "rationale_text",
            "stage_a_rhetorical_heuristics",
        ]
        tactic["implementation_status"] = "blocked"
        tactic["missing_capabilities"] = [
            "current structured evaluator consumes rendered tactic.rhetorical labels",
            "validation-selected sigmoid threshold path for tactic logits is not yet exposed",
        ]
        contract["implementation_status"] = "partially_ready"
        contract["missing_capabilities"].extend(tactic["missing_capabilities"])
    return contract


def write_metric_contract_artifact(contract: dict[str, Any], output_path: str | Path) -> Path:
    """Write a JSON metric-contract artifact."""

    path = Path(output_path)
    write_json(path, contract)
    return path


def _labels_for_field(vocab: LabelVocab, field: str) -> list[str]:
    if field in vocab.single_label_fields:
        return list(vocab.single_label_fields[field])
    if field in vocab.multi_label_fields:
        return list(vocab.multi_label_fields[field])
    if field in vocab.binary_fields:
        return [False, True]
    return []


__all__ = ["resolve_metric_contract", "write_metric_contract_artifact"]
