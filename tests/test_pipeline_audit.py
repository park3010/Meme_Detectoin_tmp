from __future__ import annotations

import json
from pathlib import Path

from experiments.pipeline_audit import audit_run_artifacts, write_audit_report


def test_passing_pipeline_artifacts_detect_logits_and_provenance(tmp_path: Path):
    run_root = _write_artifacts(tmp_path, metrics={"accuracy": 0.75, "macro_f1": 0.7})

    result = audit_run_artifacts(run_root, strict=True, require_nonempty_metrics=True)

    assert result["passed"] is True
    assert result["training_log"]["missing_expected_logits_losses"] == []
    assert result["training_log"]["auxiliary_loss_checks"]["target_presence"]["gradient_ok"] is True
    assert result["training_log"]["auxiliary_loss_checks"]["tactic_multimodal_relation"]["gradient_ok"] is True
    assert result["predictions"]["contract_pass_count"] == 1
    assert result["metrics"]["metrics_usable"] is True

    report = write_audit_report(result, run_root / "pipeline_audit_report.md")
    assert report.exists()
    assert "# Pipeline Audit Report" in report.read_text(encoding="utf-8")


def test_missing_auxiliary_logits_warns_non_strict_and_fails_strict(tmp_path: Path):
    record = _prediction_record()
    del record["target"]["presence_logits"]
    del record["training_hooks"]["target_presence_logits"]
    del record["tactic"]["multimodal_relation_logits"]
    del record["training_hooks"]["tactic_multimodal_relation_logits"]
    run_root = _write_artifacts(tmp_path, prediction=record, metrics={"accuracy": 1.0, "macro_f1": 1.0})

    advisory = audit_run_artifacts(run_root, strict=False)
    strict = audit_run_artifacts(run_root, strict=True)

    assert advisory["passed"] is True
    assert advisory["warnings"]
    assert strict["passed"] is False
    assert any("Stage E artifact contract" in error for error in strict["errors"])


def test_empty_metrics_allowance_and_requirement(tmp_path: Path):
    run_root = _write_artifacts(
        tmp_path,
        metrics={"accuracy": None, "macro_f1": None, "roc_auc": None},
        split_sizes={"train": 1, "valid": 0, "test": 0},
    )

    allowed = audit_run_artifacts(
        run_root,
        strict=True,
        require_nonempty_metrics=True,
        allow_empty_split=True,
    )
    required = audit_run_artifacts(
        run_root,
        strict=True,
        require_nonempty_metrics=True,
        allow_empty_split=False,
    )

    assert allowed["passed"] is True
    assert allowed["metrics"]["empty_split_allowed"] is True
    assert required["passed"] is False
    assert any("Non-empty metrics were required" in error for error in required["errors"])


def _write_artifacts(
    tmp_path: Path,
    *,
    prediction: dict | None = None,
    metrics: dict | None = None,
    split_sizes: dict[str, int] | None = None,
) -> Path:
    run_root = tmp_path / "run"
    run_root.mkdir()
    active_logits = [
        "harmfulness",
        "target_granularity",
        "target_presence",
        "intent_primary",
        "tactic_rhetorical",
        "tactic_multimodal_relation",
    ]
    provenance = {
        name: {
            "provenance": "logits_aux_with_proxy_fallback" if name in {"target_presence", "tactic_multimodal_relation"} else "logits",
            "differentiable_expected": True,
            "mean_requires_grad": 1.0,
        }
        for name in active_logits
    }
    training_log = [
        {
            "epoch": 1,
            "train_loss": 1.0,
            "split_sizes": split_sizes or {"train": 8, "valid": 1, "test": 2},
            "loss_components": {name: 0.1 for name in active_logits},
            "loss_provenance": provenance,
            "active_logits_losses": active_logits,
            "active_proxy_losses": ["stance"],
            "active_logits_loss_count": len(active_logits),
            "active_proxy_loss_count": 1,
            "val_accuracy": 0.5 if not split_sizes or split_sizes.get("valid") else None,
            "val_macro_f1": 0.5 if not split_sizes or split_sizes.get("valid") else None,
            "val_roc_auc": 0.5 if not split_sizes or split_sizes.get("valid") else None,
            "val_tn": 1 if not split_sizes or split_sizes.get("valid") else 0,
            "val_fp": 0,
            "val_fn": 0,
            "val_tp": 0,
        }
    ]
    (run_root / "training_log.json").write_text(json.dumps(training_log), encoding="utf-8")
    (run_root / "final_predictions.jsonl").write_text(
        json.dumps(prediction or _prediction_record()) + "\n",
        encoding="utf-8",
    )
    (run_root / "metrics.json").write_text(json.dumps(metrics or {"accuracy": 0.75, "macro_f1": 0.7}), encoding="utf-8")
    return run_root


def _prediction_record() -> dict:
    trainable_fields = [
        "harmfulness.label",
        "target.granularity",
        "target.presence",
        "intent.primary",
        "tactic.rhetorical",
        "tactic.multimodal_relation",
    ]
    field_provenance = {
        "harmfulness.label": "logits",
        "target.granularity": "logits",
        "target.presence": "logits_aux",
        "target.heuristic_presence": "heuristic_proxy",
        "intent.primary": "logits",
        "tactic.rhetorical": "logits_multilabel_or_top1_rendered",
        "tactic.multimodal_relation": "logits_aux",
        "tactic.stage_a_multimodal_relation": "stage_a_cue_proxy",
        "rationale": "template",
    }
    return {
        "sample_id": "sample-1",
        "target": {
            "presence": "explicit",
            "presence_scores": {"explicit": 0.8, "implicit": 0.1, "none": 0.1},
            "presence_logits": [2.0, 0.0, 0.0],
            "presence_source": "target_presence_head",
            "presence_provenance": "logits_aux",
            "heuristic_presence": "explicit",
            "heuristic_presence_score": 0.7,
        },
        "tactic": {
            "multimodal_relation": "cross_modal_implication",
            "multimodal_relation_scores": {"cross_modal_implication": 0.8},
            "multimodal_relation_logits": [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            "multimodal_relation_source": "tactic_multimodal_relation_head",
            "multimodal_relation_provenance": "logits_aux",
            "stage_a_multimodal_relation": "cross_modal_implication",
            "rhetorical_primary": "sarcasm_irony",
            "rhetorical_labels": ["sarcasm_irony"],
            "rhetorical_decoding": "top1_logits_plus_heuristic_cues",
            "heuristic_rhetorical_cues": ["sarcasm_irony"],
        },
        "supporting_evidence": {
            "internal": [
                {
                    "source_stage": "stage_a",
                    "modality": "text",
                    "grounding_type": "global",
                    "is_heuristic": False,
                    "attribution_backend": "gate_attention_score_proxy",
                }
            ],
            "external": [
                {
                    "candidate_origin": "retrieved",
                    "is_external_knowledge": True,
                    "is_generated": False,
                    "is_fallback": False,
                    "is_retrieved": True,
                    "verification_status": "accepted",
                    "attribution_backend": "final_score_attention_support_proxy",
                }
            ],
        },
        "output_provenance": {
            "field_provenance": field_provenance,
            "trainable_logits_fields": trainable_fields,
            "proxy_fields": ["intent.stance", "supporting_evidence"],
            "template_fields": ["rationale"],
            "cue_fields": ["target.heuristic_presence", "tactic.stage_a_multimodal_relation"],
            "label_spaces": {
                "target_presence": ["explicit", "implicit", "none"],
                "tactic_multimodal_relation": [
                    "complementary",
                    "incongruent",
                    "cross_modal_implication",
                    "text_only",
                    "image_only",
                    "none",
                    "other",
                ],
            },
            "stage_d_trace_available": True,
        },
        "training_hooks": {
            "target_presence_logits": [2.0, 0.0, 0.0],
            "target_presence_scores": {"explicit": 0.8},
            "tactic_multimodal_relation_logits": [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            "tactic_multimodal_relation_scores": {"cross_modal_implication": 0.8},
            "field_provenance": field_provenance,
            "trainable_logits_fields": trainable_fields,
            "proxy_fields": ["intent.stance", "supporting_evidence"],
            "stage_d_trace_available": True,
        },
        "stage_metadata": {
            "stage_d": {"attention_trace": {"top_external_index": 0}},
            "stage_e": {"stage_d_trace_available": True},
        },
    }
