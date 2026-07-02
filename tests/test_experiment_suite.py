from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from experiments.experiment_suite import resolve_suite_plan, run_suite


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
