"""Canonical experiment-suite planner and executor."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dataset import MemeDataset
from experiments.ablation_configs import TRAIN_TIME_CORE_ABLATIONS, component_state_for_ablation, get_ablation_contract, normalize_ablation_name
from experiments.ablation_runner import run_ablation_experiment
from experiments.knowledge_comparison import run_knowledge_comparison
from experiments.pipeline_audit import audit_run_artifacts, format_audit_summary, write_audit_report
from experiments.progress import progress_iter
from experiments.run_manifest import current_command, update_run_manifest
from experiments.splits import build_splits_for_dataset, normalize_dataset_names, save_splits
from experiments.train import BaselineRunConfig, OursRunConfig, run_baseline_experiment, run_ours_experiment
from utils.io import load_yaml, write_json


@dataclass
class SuiteRun:
    """One resolved suite run."""

    suite_name: str
    run_kind: str
    run_name: str
    dataset: str
    seed: int
    config_path: str
    split_file: str
    output_root: str
    epochs: int
    limit: int | None
    device: str
    baseline: str | None = None
    ablation: str | None = None
    knowledge_mode: str | None = None

    @property
    def output_dir(self) -> Path:
        return Path(self.output_root) / "predictions" / self.dataset / self.run_name / str(self.seed)


@dataclass
class SuitePlan:
    """Resolved suite plan."""

    name: str
    description: str
    runs: list[SuiteRun]
    suite_manifest_path: Path
    audit_after_run: bool
    require_nonempty_metrics: bool


def resolve_suite_plan(
    suite_name: str,
    *,
    config_path: str = "configs/config.yaml",
    datasets: list[str] | None = None,
    seeds: list[int] | None = None,
    epochs: int | None = None,
    limit: int | None = None,
    device: str = "cpu",
    output_root: str = "result",
    split_file: str | None = None,
    dry_run: bool = False,
) -> SuitePlan:
    """Resolve a suite preset into concrete runs without executing them."""

    cfg = load_yaml(config_path)
    suite_cfg = cfg.get("experiments", {}).get("suites", {}).get(suite_name)
    if not isinstance(suite_cfg, dict):
        raise ValueError(f"Unknown experiment suite: {suite_name}")

    resolved_datasets = normalize_dataset_names(datasets) or list(suite_cfg.get("datasets", ["harm_c"]))
    resolved_seeds = list(seeds or suite_cfg.get("seeds", [42]))
    suite_epochs = int(epochs if epochs is not None else _suite_epochs(suite_cfg, cfg))
    suite_limit = limit if limit is not None else suite_cfg.get("limit")
    if suite_limit is not None:
        suite_limit = int(suite_limit)

    runs: list[SuiteRun] = []
    include = suite_cfg.get("include", {}) or {}
    for dataset in resolved_datasets:
        for seed in resolved_seeds:
            split_path = _split_path(dataset, seed, split_file, output_root)
            if include.get("full_model", False):
                runs.append(
                    _suite_run(
                        suite_name,
                        "ours_full",
                        "ours_full",
                        dataset,
                        seed,
                        config_path,
                        split_path,
                        output_root,
                        suite_epochs,
                        suite_limit,
                        device,
                    )
                )
            for baseline in include.get("baselines", []) or []:
                runs.append(
                    _suite_run(
                        suite_name,
                        "baseline",
                        str(baseline),
                        dataset,
                        seed,
                        config_path,
                        split_path,
                        output_root,
                        suite_epochs,
                        suite_limit,
                        device,
                        baseline=str(baseline),
                    )
                )
            for raw_ablation in include.get("ablations", []) or []:
                ablation = normalize_ablation_name(str(raw_ablation))
                contract = get_ablation_contract(ablation)
                if not contract.supported:
                    continue
                runs.append(
                    _suite_run(
                        suite_name,
                        "ablation",
                        f"ablation_{ablation}",
                        dataset,
                        seed,
                        config_path,
                        split_path,
                        output_root,
                        suite_epochs,
                        suite_limit,
                        device,
                        ablation=ablation,
                    )
                )
            for mode in include.get("knowledge_modes", []) or []:
                runs.append(
                    _suite_run(
                        suite_name,
                        "knowledge_comparison",
                        f"knowledge_{mode}",
                        dataset,
                        seed,
                        config_path,
                        split_path,
                        output_root,
                        suite_epochs,
                        suite_limit,
                        device,
                        knowledge_mode=str(mode),
                    )
                )

    return SuitePlan(
        name=suite_name,
        description=str(suite_cfg.get("description", "")),
        runs=runs,
        suite_manifest_path=Path(output_root) / "experiment_suites" / suite_name / "suite_manifest.json",
        audit_after_run=bool(suite_cfg.get("audit_after_run", False)),
        require_nonempty_metrics=bool(suite_cfg.get("require_nonempty_metrics", False)),
    )


def run_suite(args: Any) -> dict[str, Any]:
    """Execute one suite from parsed CLI args."""

    plan = resolve_suite_plan(
        args.suite,
        config_path=args.config,
        datasets=args.dataset,
        seeds=args.seed,
        epochs=args.epochs,
        limit=args.limit,
        device=args.device,
        output_root=args.output_root,
        split_file=args.split_file,
        dry_run=args.dry_run,
    )
    _print_plan(plan)
    if args.dry_run:
        return {"dry_run": True, "planned_runs": len(plan.runs)}

    manifest = _suite_manifest(plan, args, status="running")
    write_json(plan.suite_manifest_path, manifest)
    statuses: list[dict[str, Any]] = []
    run_iter = progress_iter(plan.runs, desc=f"suite {plan.name}", disable=args.disable_tqdm)
    for index, run in enumerate(run_iter, start=1):
        print(f"[suite:{plan.name}] [{index}/{len(plan.runs)}] {run.dataset} seed={run.seed} {run.run_name}")
        _ensure_split(run, args)
        if (args.resume or args.skip_complete) and _is_complete(run, args):
            statuses.append(_status(run, "skipped_complete"))
            continue
        try:
            metrics = _execute_run(run, args)
            audit = None
            if (args.audit_after_run or plan.audit_after_run) and run.run_kind in {"ours_full", "ablation"}:
                audit = _audit_run(run, args, plan)
                update_run_manifest(run.output_dir, {"audit": _compact_audit(audit), "last_audit_passed": bool(audit.get("passed"))})
                if args.strict and audit.get("errors"):
                    raise RuntimeError(f"Audit failed for {run.run_name}: {audit.get('errors')}")
            statuses.append(_status(run, "complete", metrics=metrics, audit=audit))
        except Exception as exc:  # pragma: no cover - exercised by CLI failure paths.
            statuses.append(_status(run, "failed", error=str(exc)))
            manifest["runs"] = statuses
            manifest["status"] = "failed"
            manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(plan.suite_manifest_path, manifest)
            raise

        manifest["runs"] = statuses
        manifest["completed_run_count"] = len([row for row in statuses if row["status"] in {"complete", "skipped_complete"}])
        write_json(plan.suite_manifest_path, manifest)

    manifest["status"] = "complete"
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["runs"] = statuses
    manifest["completed_run_count"] = len([row for row in statuses if row["status"] in {"complete", "skipped_complete"}])
    write_json(plan.suite_manifest_path, manifest)
    print(f"Suite manifest: {plan.suite_manifest_path}")
    return manifest


def _execute_run(run: SuiteRun, args: Any) -> dict[str, Any]:
    command = current_command()
    if run.run_kind == "ours_full":
        return run_ours_experiment(
            OursRunConfig(
                dataset_name=run.dataset,
                seed=run.seed,
                config_path=run.config_path,
                split_file=run.split_file,
                output_root=run.output_root,
                model_name=run.run_name,
                epochs=run.epochs,
                lr=args.lr,
                patience=args.patience,
                min_delta=args.min_delta,
                early_stop_metric=args.early_stop_metric,
                disable_tqdm=args.disable_tqdm,
                print_components=args.print_components,
                device=run.device,
                limit=run.limit,
                suite_name=run.suite_name,
                requested_command=command,
            )
        )
    if run.run_kind == "baseline":
        return run_baseline_experiment(
            BaselineRunConfig(
                model_name=run.baseline or run.run_name,
                dataset_name=run.dataset,
                seed=run.seed,
                config_path=run.config_path,
                split_file=run.split_file,
                output_root=run.output_root,
                epochs=run.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                patience=args.patience,
                min_delta=args.min_delta,
                early_stop_metric=args.early_stop_metric,
                disable_tqdm=args.disable_tqdm,
                device=run.device,
                limit=run.limit,
                suite_name=run.suite_name,
                requested_command=command,
            )
        )
    if run.run_kind == "ablation":
        if run.ablation in TRAIN_TIME_CORE_ABLATIONS:
            return run_ours_experiment(
                OursRunConfig(
                    dataset_name=run.dataset,
                    seed=run.seed,
                    config_path=run.config_path,
                    split_file=run.split_file,
                    output_root=run.output_root,
                    model_name=run.run_name,
                    epochs=run.epochs,
                    lr=args.lr,
                    patience=args.patience,
                    min_delta=args.min_delta,
                    early_stop_metric=args.early_stop_metric,
                    disable_tqdm=args.disable_tqdm,
                    print_components=args.print_components,
                    device=run.device,
                    limit=run.limit,
                    ablation_name=run.ablation,
                    structured_auxiliary=(run.ablation != "w_o_structured_auxiliary"),
                    suite_name=run.suite_name,
                    requested_command=command,
                )
            )
        return run_ablation_experiment(
            run.dataset,
            run.ablation or "full",
            seed=run.seed,
            config_path=run.config_path,
            split_file=run.split_file,
            output_root=run.output_root,
            limit=run.limit,
            disable_tqdm=args.disable_tqdm,
            print_components=args.print_components,
            device=run.device,
            suite_name=run.suite_name,
            requested_command=command,
        )
    if run.run_kind == "knowledge_comparison":
        return run_knowledge_comparison(
            run.dataset,
            run.knowledge_mode or "verified",
            seed=run.seed,
            config_path=run.config_path,
            split_file=run.split_file,
            output_root=run.output_root,
            limit=run.limit,
            disable_tqdm=args.disable_tqdm,
            print_components=args.print_components,
            device=run.device,
            suite_name=run.suite_name,
            requested_command=command,
        )
    raise ValueError(f"Unsupported suite run kind: {run.run_kind}")


def _ensure_split(run: SuiteRun, args: Any) -> Path:
    path = Path(run.split_file)
    if args.split_file or path.exists():
        return path
    cfg = load_yaml(run.config_path)
    dataset = MemeDataset(
        dataset_root=cfg.get("paths", {}).get("dataset_root", "dataset/source"),
        annotation_root=cfg.get("paths", {}).get("annotation_root", "dataset/annotation"),
        dataset_names=[run.dataset],
        keep_missing_images=True,
        limit=run.limit,
    )
    splits = build_splits_for_dataset(
        run.dataset,
        dataset,
        seed=run.seed,
        dataset_root=cfg.get("paths", {}).get("dataset_root", "dataset/source"),
    )
    return save_splits(splits, run.dataset, run.seed, Path(run.output_root) / "splits")


def _audit_run(run: SuiteRun, args: Any, plan: SuitePlan) -> dict[str, Any]:
    result = audit_run_artifacts(
        run.output_dir,
        require_nonempty_metrics=args.require_nonempty_metrics or plan.require_nonempty_metrics,
        strict=args.strict,
    )
    write_audit_report(result, run.output_dir / "pipeline_audit_report.md")
    print(format_audit_summary(result))
    return result


def _is_complete(run: SuiteRun, args: Any) -> bool:
    manifest_path = run.output_dir / "run_manifest.json"
    metrics_path = run.output_dir / "metrics.json"
    predictions_path = run.output_dir / "final_predictions.jsonl"
    if not (manifest_path.exists() and metrics_path.exists() and predictions_path.exists()):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if run.run_kind in {"ours_full", "ablation"}:
        audit = audit_run_artifacts(run.output_dir, require_nonempty_metrics=args.require_nonempty_metrics, strict=args.strict)
        return bool(audit.get("passed"))
    return manifest.get("completion_status") == "complete"


def _suite_run(
    suite_name: str,
    run_kind: str,
    run_name: str,
    dataset: str,
    seed: int,
    config_path: str,
    split_path: Path,
    output_root: str,
    epochs: int,
    limit: int | None,
    device: str,
    **kwargs: Any,
) -> SuiteRun:
    return SuiteRun(
        suite_name=suite_name,
        run_kind=run_kind,
        run_name=run_name,
        dataset=dataset,
        seed=int(seed),
        config_path=str(config_path),
        split_file=str(split_path),
        output_root=str(output_root),
        epochs=int(epochs),
        limit=limit,
        device=device,
        **kwargs,
    )


def _split_path(dataset: str, seed: int, split_file: str | None, output_root: str) -> Path:
    if split_file:
        return Path(split_file)
    return Path(output_root) / "splits" / dataset / f"seed_{seed}.json"


def _suite_epochs(suite_cfg: dict[str, Any], cfg: dict[str, Any]) -> int:
    if suite_cfg.get("epochs") is not None:
        return int(suite_cfg["epochs"])
    return int(cfg.get("experiments", {}).get("ours_full", {}).get("epochs", 5))


def _suite_manifest(plan: SuitePlan, args: Any, status: str) -> dict[str, Any]:
    return {
        "schema": "experiment_suite_manifest_v1",
        "suite_name": plan.name,
        "description": plan.description,
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": args.config,
        "requested_command": current_command(),
        "run_count": len(plan.runs),
        "completed_run_count": 0,
        "runs_planned": [asdict(run) | {"output_dir": str(run.output_dir)} for run in plan.runs],
        "runs": [],
    }


def _status(run: SuiteRun, status: str, metrics: dict[str, Any] | None = None, audit: dict[str, Any] | None = None, error: str | None = None) -> dict[str, Any]:
    row = {
        "run_kind": run.run_kind,
        "run_name": run.run_name,
        "dataset": run.dataset,
        "seed": run.seed,
        "status": status,
        "output_dir": str(run.output_dir),
    }
    if metrics is not None:
        row["metrics"] = {"accuracy": metrics.get("accuracy"), "macro_f1": metrics.get("macro_f1")}
    if audit is not None:
        row["audit"] = _compact_audit(audit)
    if error:
        row["error"] = error
    return row


def _compact_audit(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": bool(audit.get("passed")),
        "status": audit.get("status"),
        "warning_count": len(audit.get("warnings", [])),
        "error_count": len(audit.get("errors", [])),
    }


def _print_plan(plan: SuitePlan) -> None:
    print(f"Experiment suite: {plan.name}")
    print(f"Description: {plan.description}")
    print(f"Runs planned: {len(plan.runs)}")
    for index, run in enumerate(plan.runs, start=1):
        details = f"{run.run_kind}:{run.run_name} dataset={run.dataset} seed={run.seed} split={run.split_file}"
        if run.ablation:
            details += f" component_state={component_state_for_ablation(run.ablation)}"
        print(f"  [{index:02d}] {details}")
