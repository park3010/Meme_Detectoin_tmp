from __future__ import annotations

import json

from experiments.evaluation import attach_formal_tactic_traces, evaluate_tactic_rhetorical_logits_only
from experiments.tactic_decoding import (
    TacticDecodingSpec,
    canonicalize_gold_tactic_labels,
    compute_tactic_logits_only_metrics,
    decode_tactic_logits,
    select_tactic_threshold,
    sigmoid_probabilities,
)


def _spec(thresholds: list[float] | None = None) -> TacticDecodingSpec:
    return TacticDecodingSpec(
        schema_version="test",
        label_order=["sarcasm_irony", "stereotype", "none"],
        non_none_labels=["sarcasm_irony", "stereotype"],
        none_label="none",
        prediction_source="tactic_logits_sigmoid",
        threshold_policy="validation_grid_search",
        threshold_candidates=thresholds or [0.5],
    )


def test_sigmoid_and_non_none_decoding_never_thresholds_none():
    spec = _spec()
    probs = sigmoid_probabilities([0.0, 2.0, 20.0])
    assert len(probs) == 3
    decoded = decode_tactic_logits([-5.0, -5.0, 20.0], spec, threshold=0.5)
    assert decoded["predicted_labels_with_none_fallback"] == ["none"]
    assert decoded["predicted_none"] is True

    decoded = decode_tactic_logits([3.0, -5.0, 20.0], spec, threshold=0.5)
    assert decoded["predicted_non_none_labels"] == ["sarcasm_irony"]
    assert decoded["predicted_labels_with_none_fallback"] == ["sarcasm_irony"]
    assert "none" not in decoded["predicted_labels_with_none_fallback"]
    assert decoded["rendered_labels_used"] is False


def test_gold_canonicalization_none_conflicts_and_unknown():
    spec = _spec()
    labels, diag = canonicalize_gold_tactic_labels(["none"], spec)
    assert labels == set()
    assert diag["gold_none"] is True

    labels, diag = canonicalize_gold_tactic_labels(["none", "sarcasm_irony"], spec)
    assert labels == {"sarcasm_irony"}
    assert diag["none_conflict_removed"] is True

    labels, diag = canonicalize_gold_tactic_labels(["unknown"], spec)
    assert labels == set()
    assert diag["usable"] is False
    assert diag["ignored_labels"] == ["unknown"]


def test_validation_threshold_selection_and_tie_breaking():
    spec = _spec([0.4, 0.7])
    result = select_tactic_threshold(
        validation_logits=[[1.0, -2.0, 0.0], [0.0, -2.0, 0.0]],
        validation_gold_labels=[["sarcasm_irony"], ["none"]],
        spec=spec,
    )
    assert result.selected_threshold == 0.7

    tie = select_tactic_threshold(
        validation_logits=[[-2.0, -2.0, 0.0]],
        validation_gold_labels=[["unknown"]],
        spec=_spec([0.4, 0.5, 0.6]),
    )
    assert tie.selected_threshold == 0.5


def test_formal_metrics_are_json_serializable_and_split_none_metric():
    spec = _spec([0.5])
    metrics = compute_tactic_logits_only_metrics(
        logits=[[3.0, -3.0, 0.0], [-3.0, -3.0, 9.0]],
        gold_labels=[["sarcasm_irony"], ["none"]],
        spec=spec,
        threshold=0.5,
    )
    assert metrics["tactic_rhetorical_formal_status"] == "ready"
    assert metrics["tactic_rhetorical_macro_f1_logits_only"] == 0.5
    assert metrics["tactic_rhetorical_micro_f1_logits_only"] == 1.0
    assert metrics["tactic_rhetorical_none_f1"] == 1.0
    assert metrics["tactic_rhetorical_exact_match_ratio"] == 1.0
    json.dumps(metrics)


def test_evaluation_ignores_rendered_rhetorical_fields():
    validation = [
        {
            "sample_id": "v1",
            "tactic_rhetorical_logits": [3.0, -3.0, 0.0],
            "tactic_rhetorical_label_order": ["sarcasm_irony", "stereotype", "none"],
            "gold_tactic": {"tactic_rhetorical": ["sarcasm_irony"]},
            "tactic": {"rhetorical": ["stereotype"]},
        }
    ]
    test = [
        {
            "sample_id": "t1",
            "tactic_rhetorical_logits": [3.0, -3.0, 0.0],
            "tactic_rhetorical_label_order": ["sarcasm_irony", "stereotype", "none"],
            "gold_tactic": {"tactic_rhetorical": ["sarcasm_irony"]},
            "tactic": {"rhetorical": ["stereotype"]},
        }
    ]
    metrics = evaluate_tactic_rhetorical_logits_only(test, threshold_selection_records=validation)
    assert metrics["tactic_rhetorical_formal_status"] == "ready"
    assert metrics["tactic_rhetorical_macro_f1_logits_only"] == 0.5
    assert metrics["tactic_rhetorical_rendered_labels_used"] is False

    spec = _spec([metrics["tactic_rhetorical_validation_selected_threshold"]])
    traced = attach_formal_tactic_traces(test, spec, metrics["tactic_rhetorical_validation_selected_threshold"])
    trace = traced[0]["evaluation"]["tactic_rhetorical_formal"]
    assert trace["predicted_non_none_labels"] == ["sarcasm_irony"]
    assert trace["rendered_labels_used"] is False
