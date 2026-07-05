from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from experiments.experiment_suite import SuiteRun, _execute_run, resolve_suite_plan, run_suite


def test_core_smoke_suite_resolves_expected_runs(tmp_path: Path):
    plan = resolve_suite_plan(
        "core_smoke",
        config_path="configs/config.yaml",
        output_root=str(tmp_path / "result"),
        device="cpu",
    )

    names = [run.run_name for run in plan.runs]
    assert names == [
        "ours_full",
        "text_only_encoder",
        "ablation_w_o_retrieval",
        "ablation_w_o_task_aware_gate",
        "ablation_w_o_structured_auxiliary",
    ]
    assert len({run.split_file for run in plan.runs}) == 1
    assert plan.suite_manifest_path == tmp_path / "result" / "experiment_suites" / "core_smoke" / "suite_manifest.json"


def test_suite_overrides_dataset_seed_and_limit(tmp_path: Path):
    plan = resolve_suite_plan(
        "core_1seed",
        config_path="configs/config.yaml",
        datasets=["harm_p"],
        seeds=[52],
        epochs=2,
        limit=7,
        output_root=str(tmp_path / "result"),
        device="cpu",
    )

    assert {run.dataset for run in plan.runs} == {"harm_p"}
    assert {run.seed for run in plan.runs} == {52}
    assert {run.epochs for run in plan.runs} == {2}
    assert {run.limit for run in plan.runs} == {7}


def test_suite_dry_run_does_not_write_manifest(tmp_path: Path):
    args = Namespace(
        suite="core_smoke",
        config="configs/config.yaml",
        dataset=None,
        seed=None,
        epochs=None,
        limit=3,
        device="cpu",
        output_root=str(tmp_path / "result"),
        split_file=None,
        dry_run=True,
        resume=False,
        skip_complete=False,
        audit_after_run=False,
        strict=False,
        require_nonempty_metrics=False,
        disable_tqdm=True,
        print_components=False,
        batch_size=4,
        lr=1e-4,
        patience=1,
        min_delta=0.0,
        early_stop_metric="val_macro_f1",
    )

    result = run_suite(args)

    assert result["dry_run"] is True
    assert result["planned_runs"] == 5
    assert not (tmp_path / "result" / "experiment_suites" / "core_smoke" / "suite_manifest.json").exists()


def test_core_ablations_route_through_train_time_runner(monkeypatch, tmp_path: Path):
    calls = []

    def fake_ours(config):
        calls.append(config)
        return {"macro_f1": 1.0}

    def fail_diagnostic(*args, **kwargs):
        raise AssertionError("core ablation should not use diagnostic ablation runner")

    monkeypatch.setattr("experiments.experiment_suite.run_ours_experiment", fake_ours)
    monkeypatch.setattr("experiments.experiment_suite.run_ablation_experiment", fail_diagnostic)
    args = _args()

    for ablation in ["w_o_retrieval", "w_o_support_verifier", "w_o_task_aware_gate", "w_o_structured_auxiliary"]:
        run = _run(tmp_path, ablation=ablation)
        _execute_run(run, args)

    assert [call.ablation_name for call in calls] == [
        "w_o_retrieval",
        "w_o_support_verifier",
        "w_o_task_aware_gate",
        "w_o_structured_auxiliary",
    ]
    assert [call.model_name for call in calls] == [
        "ablation_w_o_retrieval",
        "ablation_w_o_support_verifier",
        "ablation_w_o_task_aware_gate",
        "ablation_w_o_structured_auxiliary",
    ]
    assert calls[-1].structured_auxiliary is False
    assert all(call.device == "cuda:0" for call in calls)


def test_diagnostic_ablation_and_knowledge_receive_device(monkeypatch, tmp_path: Path):
    received = {}

    def fake_ablation(*args, **kwargs):
        received["ablation_device"] = kwargs.get("device")
        return {"macro_f1": 1.0}

    def fake_knowledge(*args, **kwargs):
        received["knowledge_device"] = kwargs.get("device")
        return {"macro_f1": 1.0}

    monkeypatch.setattr("experiments.experiment_suite.run_ablation_experiment", fake_ablation)
    monkeypatch.setattr("experiments.experiment_suite.run_knowledge_comparison", fake_knowledge)

    _execute_run(_run(tmp_path, ablation="w_o_roi"), _args())
    _execute_run(_run(tmp_path, kind="knowledge_comparison", name="knowledge_verified", knowledge_mode="verified"), _args())

    assert received == {"ablation_device": "cuda:0", "knowledge_device": "cuda:0"}


def _args() -> Namespace:
    return Namespace(
        lr=1e-4,
        patience=1,
        min_delta=0.0,
        early_stop_metric="val_macro_f1",
        disable_tqdm=True,
        print_components=False,
        batch_size=4,
    )


def _run(
    tmp_path: Path,
    *,
    kind: str = "ablation",
    name: str | None = None,
    ablation: str | None = None,
    knowledge_mode: str | None = None,
) -> SuiteRun:
    run_name = name or f"ablation_{ablation}"
    return SuiteRun(
        suite_name="test_suite",
        run_kind=kind,
        run_name=run_name,
        dataset="harm_c",
        seed=42,
        config_path="configs/config.yaml",
        split_file=str(tmp_path / "split.json"),
        output_root=str(tmp_path / "result"),
        epochs=1,
        limit=2,
        device="cuda:0",
        ablation=ablation,
        knowledge_mode=knowledge_mode,
    )
