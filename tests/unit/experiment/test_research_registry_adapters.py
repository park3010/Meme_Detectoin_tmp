from __future__ import annotations

import json

import pytest
import torch

from experiments.adapters import BlockedExternalAdapter, RunContext, create_adapter
from experiments.pipeline_audit import audit_baseline_run_artifacts
from experiments.research_orchestration import (
    canonical_run_complete,
    execute_research_suite,
    plan_research,
    run_research_preflight,
)
from experiments.research_schemas import REQUIRED_RUN_ARTIFACTS
from experiments.registry import experiment_specs, load_experiment_registry, resolve_research_suite, validate_registry
from module.baseline import OpenCLIPMultimodalClassifier


OURS_FRAMEWORK_EXPERIMENTS = [
    "ours_full",
    "ablation_w_o_retrieval",
    "ablation_w_o_support_verifier",
    "ablation_w_o_task_aware_gate",
    "ablation_w_o_structured_auxiliary",
]

LEGACY_RESEARCH_SUITES = {
    "harmeme_to_fhm_smoke": {
        "description": "Small protocol-valid built-in dry-run/smoke suite.",
        "experiments": ["text_only_deberta", "ours_full"],
        "seeds": [42],
        "datasets": ["harm_c", "harm_p", "facebook"],
        "limit": 100,
        "enabled": True,
    },
    "harmeme_to_fhm_1seed": {
        "description": "One-seed built-in baseline, Ours Full, and core-ablation suite.",
        "experiments": [
            "text_only_deberta",
            "image_only_openclip",
            "image_text_concat",
            "openclip_classifier",
            *OURS_FRAMEWORK_EXPERIMENTS,
        ],
        "seeds": [42],
        "datasets": ["harm_c", "harm_p", "facebook"],
        "enabled": True,
    },
    "harmeme_to_fhm_5seed": {
        "description": "Five-seed paper suite for built-in models and core ablations.",
        "experiments": [
            "text_only_deberta",
            "image_only_openclip",
            "image_text_concat",
            "openclip_classifier",
            *OURS_FRAMEWORK_EXPERIMENTS,
        ],
        "seeds": [42, 52, 123, 777, 2026],
        "datasets": ["harm_c", "harm_p", "facebook"],
        "enabled": True,
    },
    "harmeme_to_fhm_knowledge_1seed": {
        "description": "Registered knowledge comparison; blocked until train-time modes exist.",
        "experiments": [
            "knowledge_no_knowledge",
            "knowledge_retrieved_only",
            "knowledge_generated_retrieved",
            "knowledge_verified",
        ],
        "seeds": [42],
        "datasets": ["harm_c", "harm_p", "facebook"],
        "enabled": True,
    },
    "harmeme_to_fhm_structured_1seed": {
        "description": "FHM silver structured evaluation for applicable trained models.",
        "experiments": OURS_FRAMEWORK_EXPERIMENTS,
        "seeds": [42],
        "datasets": ["facebook"],
        "enabled": True,
    },
    "harmeme_to_fhm_error_export": {
        "description": "Error and human-review package export from completed FHM runs.",
        "experiments": ["ours_full"],
        "seeds": [42],
        "datasets": ["facebook"],
        "enabled": True,
    },
}


def test_registry_is_complete_and_fhm_never_trains():
    registry = load_experiment_registry()
    assert validate_registry(registry)["passed"] is True
    for spec in experiment_specs(registry).values():
        assert "facebook" not in spec.source_train_datasets
        assert "facebook" not in spec.source_validation_datasets
        assert "memotion" not in spec.source_train_datasets
    suite = resolve_research_suite(registry, "harmeme_to_fhm_1seed")
    assert {row["experiment_id"] for row in suite["runs"]} >= {"ours_full", "openclip_classifier"}


def test_ours_framework_suites_resolve_exact_conditions_seeds_and_smoke_policy():
    registry = load_experiment_registry()
    specs = experiment_specs(registry)
    canonical_smoke = registry["suites"]["harmeme_to_fhm_smoke"]

    smoke = resolve_research_suite(registry, "ours_framework_smoke")
    assert [row["experiment_id"] for row in smoke["runs"]] == OURS_FRAMEWORK_EXPERIMENTS
    assert [row["seed"] for row in smoke["runs"]] == [42] * 5
    assert smoke["limit"] == canonical_smoke["limit"] == 100
    assert "epochs" not in registry["suites"]["ours_framework_smoke"]
    assert {specs[row["experiment_id"]].adapter for row in smoke["runs"]} == {"builtin_framework"}
    assert {specs[row["experiment_id"]].group for row in smoke["runs"]} == {"proposed", "core_ablation"}

    one_seed = resolve_research_suite(registry, "ours_framework_1seed")
    assert len(one_seed["runs"]) == 5
    assert [row["experiment_id"] for row in one_seed["runs"]] == OURS_FRAMEWORK_EXPERIMENTS
    assert [row["seed"] for row in one_seed["runs"]] == [42] * 5
    assert one_seed["limit"] is None
    assert not {"epochs", "limit"} & registry["suites"]["ours_framework_1seed"].keys()

    five_seed = resolve_research_suite(registry, "ours_framework_5seed")
    assert len(five_seed["runs"]) == 25
    assert five_seed["limit"] is None
    assert not {"epochs", "limit"} & registry["suites"]["ours_framework_5seed"].keys()
    expected_seeds = [42, 52, 123, 777, 2026]
    for offset, experiment_id in enumerate(OURS_FRAMEWORK_EXPERIMENTS):
        block = five_seed["runs"][offset * 5 : (offset + 1) * 5]
        assert [row["experiment_id"] for row in block] == [experiment_id] * 5
        assert [row["seed"] for row in block] == expected_seeds


def test_ours_framework_suites_exclude_baselines_external_and_knowledge_conditions():
    registry = load_experiment_registry()
    specs = experiment_specs(registry)
    for suite_name in ("ours_framework_smoke", "ours_framework_1seed", "ours_framework_5seed"):
        resolved = resolve_research_suite(registry, suite_name)
        assert list(dict.fromkeys(row["experiment_id"] for row in resolved["runs"])) == OURS_FRAMEWORK_EXPERIMENTS
        assert all(specs[row["experiment_id"]].adapter == "builtin_framework" for row in resolved["runs"])
        assert all(specs[row["experiment_id"]].group not in {"built_in_baseline", "knowledge_comparison"} for row in resolved["runs"])


def test_existing_research_suites_remain_unchanged():
    suites = load_experiment_registry()["suites"]
    assert {name: suites[name] for name in LEGACY_RESEARCH_SUITES} == LEGACY_RESEARCH_SUITES


def test_unknown_research_suite_still_fails_clearly():
    with pytest.raises(ValueError, match=r"^Unknown research suite: does_not_exist$"):
        resolve_research_suite(load_experiment_registry(), "does_not_exist")


def test_framework_smoke_dry_run_writes_no_training_artifacts(tmp_path):
    output_root = tmp_path / "ours_smoke"
    result = execute_research_suite(
        "ours_framework_smoke",
        output_root=str(output_root),
        execute=False,
    )

    assert len(result["runs"]) == 5
    assert {row["action"] for row in result["runs"]} == {"planned"}
    assert (output_root / "research_suites" / "ours_framework_smoke" / "suite_manifest.json").exists()
    assert not (output_root / "research_runs").exists()
    training_artifacts = {"best_model.pt", "last_model.pt", "training_log.json", "test_predictions.jsonl"}
    assert not any(path.name in training_artifacts for path in output_root.rglob("*"))


def test_registry_plan_reports_all_ours_framework_suites(tmp_path):
    plan = plan_research(output_root=str(tmp_path))
    suites = plan["registry"]["suites"]
    assert suites["ours_framework_smoke"] == {"enabled": True, "experiment_count": 5, "seeds": [42]}
    assert suites["ours_framework_1seed"] == {"enabled": True, "experiment_count": 5, "seeds": [42]}
    assert suites["ours_framework_5seed"] == {
        "enabled": True,
        "experiment_count": 5,
        "seeds": [42, 52, 123, 777, 2026],
    }


def test_research_strict_preflight_still_passes(tmp_path):
    result = run_research_preflight(output_root=str(tmp_path), strict=True)
    assert result["passed"] is True
    assert result["status"] == "pass"


def test_blocked_external_adapter_cannot_execute(tmp_path):
    specs = experiment_specs(load_experiment_registry())
    adapter = create_adapter(specs["gpt4o_direct"], RunContext(suite="test", seed=42, output_root=str(tmp_path)))
    assert isinstance(adapter, BlockedExternalAdapter)
    assert adapter.prepare_data()["status"] == "blocked_api_credentials"
    with pytest.raises(RuntimeError):
        adapter.train_or_fit()


def test_run_context_exposes_partial_run_resume_policy(tmp_path):
    context = RunContext(suite="test", seed=42, output_root=str(tmp_path), resume=True)
    assert context.resume is True


def test_shared_openclip_baseline_has_distinct_interaction_classifier():
    model = OpenCLIPMultimodalClassifier(hidden_dim=32, prefer_pretrained_clip=False)
    output = model([None, None], ["first meme", "second meme"])
    assert output["logits"].shape == (2, 2)
    assert output["prob_harmful"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()


def test_canonical_completion_requires_every_artifact_and_passing_audit(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for name in REQUIRED_RUN_ARTIFACTS:
        (run_dir / name).write_text("{}" if name.endswith(".json") else "", encoding="utf-8")
    (run_dir / "run_manifest.json").write_text('{"completion_status":"complete"}', encoding="utf-8")
    (run_dir / "pipeline_audit_report.json").write_text('{"passed":true}', encoding="utf-8")
    assert canonical_run_complete(run_dir) is True
    (run_dir / "metrics.json").unlink()
    assert canonical_run_complete(run_dir) is False


def test_baseline_audit_uses_harmfulness_contract_not_stage_e_contract(tmp_path):
    run_dir = tmp_path / "baseline"
    run_dir.mkdir()
    manifest = {
        "run_kind": "baseline",
        "source_train_manifest_sha256": "source",
        "source_validation_manifest_sha256": "source",
        "fhm_test_manifest_sha256": "fhm",
        "threshold_selection_dataset": "HarMeme validation",
        "heldout_test_dataset": "facebook",
    }
    prediction = {
        "sample_id": "1",
        "dataset_name": "facebook",
        "gold_label": 1,
        "pred_label": 1,
        "prob_harmful": 0.8,
        "logits": [0.1, 0.9],
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "training_log.json").write_text(json.dumps([{"epoch": 1}]), encoding="utf-8")
    line = json.dumps(prediction) + "\n"
    (run_dir / "validation_predictions.jsonl").write_text(line, encoding="utf-8")
    (run_dir / "final_predictions.jsonl").write_text(line, encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps({"accuracy": 1.0, "macro_f1": 1.0}), encoding="utf-8")

    result = audit_baseline_run_artifacts(run_dir, strict=True)
    assert result["passed"] is True
    assert result["audit_contract"] == "harmfulness_baseline_v1"
