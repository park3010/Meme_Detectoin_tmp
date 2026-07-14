from __future__ import annotations

import json

import pytest
import torch

from experiments.adapters import BlockedExternalAdapter, RunContext, create_adapter
from experiments.pipeline_audit import audit_baseline_run_artifacts
from experiments.research_orchestration import canonical_run_complete
from experiments.research_schemas import REQUIRED_RUN_ARTIFACTS
from experiments.registry import experiment_specs, load_experiment_registry, resolve_research_suite, validate_registry
from module.baseline import OpenCLIPMultimodalClassifier


def test_registry_is_complete_and_fhm_never_trains():
    registry = load_experiment_registry()
    assert validate_registry(registry)["passed"] is True
    for spec in experiment_specs(registry).values():
        assert "facebook" not in spec.source_train_datasets
        assert "facebook" not in spec.source_validation_datasets
        assert "memotion" not in spec.source_train_datasets
    suite = resolve_research_suite(registry, "harmeme_to_fhm_1seed")
    assert {row["experiment_id"] for row in suite["runs"]} >= {"ours_full", "openclip_classifier"}


def test_blocked_external_adapter_cannot_execute(tmp_path):
    specs = experiment_specs(load_experiment_registry())
    adapter = create_adapter(specs["gpt4o_direct"], RunContext(suite="test", seed=42, output_root=str(tmp_path)))
    assert isinstance(adapter, BlockedExternalAdapter)
    assert adapter.prepare_data()["status"] == "blocked_api_credentials"
    with pytest.raises(RuntimeError):
        adapter.train_or_fit()


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
