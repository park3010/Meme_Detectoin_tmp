"""Unified command-line entry point for framework training and evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import ROOT  # noqa: F401 - importing common also adds the repository root to sys.path.

from experiments.ablation_configs import ABLATION_MODES
from experiments.ablation_runner import run_ablation_experiment, run_fusion_experiment
from experiments.evaluation import evaluate_prediction_file
from experiments.pipeline_audit import audit_run_artifacts, format_audit_summary, write_audit_report
from experiments.experiment_suite import run_suite
from experiments.preflight import format_preflight_summary, run_preflight
from experiments.splits import DEFAULT_SEEDS, normalize_dataset_names
from experiments.train import BaselineRunConfig, OursRunConfig, run_baseline_experiment, run_ours_experiment
from module.runner import PipelineRunner


DEFAULT_CONFIG = str(ROOT / "configs" / "config.yaml")


def build_parser() -> argparse.ArgumentParser:
    """Build the unified CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train/evaluate the full proposed framework.")
    _add_ours_args(train)
    train.set_defaults(func=_cmd_train)

    baseline = subparsers.add_parser("baseline", help="Train/evaluate a simple baseline.")
    _add_baseline_args(baseline)
    baseline.set_defaults(func=_cmd_baseline)

    stage = subparsers.add_parser("stage", help="Run the pipeline through a selected stage.")
    stage.add_argument("--config", default=DEFAULT_CONFIG)
    stage.add_argument("--dataset", nargs="+", default=["harm_c"])
    stage.add_argument("--until", choices=["stage_a", "stage_b", "stage_c", "stage_d", "stage_e", "a", "b", "c", "d", "e"], default="stage_e")
    stage.add_argument("--limit", type=int, default=None)
    stage.add_argument("--device", default=None)
    stage.add_argument("--no-save", action="store_true")
    stage.set_defaults(func=_cmd_stage)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate structured prediction outputs.")
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--model", default="ours_full")
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--prediction-file", default=None)
    evaluate.add_argument("--result-root", default="result")
    evaluate.add_argument("--output-root", default="result/metrics")
    evaluate.add_argument("--disable-tqdm", action="store_true")
    evaluate.set_defaults(func=_cmd_evaluate)

    ablation = subparsers.add_parser("ablation", help="Run one stage-wise ablation or fusion mode.")
    ablation.add_argument("--config", default=DEFAULT_CONFIG)
    ablation.add_argument("--dataset", nargs="+", default=["harm_c"])
    ablation.add_argument("--experiment", default=None, help="Ablation preset such as ablations.w_o_retrieval.")
    ablation.add_argument("--ablation", nargs="+", default=None, help="Ablation name(s), or all.")
    ablation.add_argument("--fusion-mode", nargs="+", default=None, help="Optional fusion mode(s), or all.")
    ablation.add_argument("--seed", nargs="+", type=int, default=[42])
    ablation.add_argument("--all-seeds", action="store_true")
    ablation.add_argument("--limit", type=int, default=None)
    ablation.add_argument("--split-file", default=None)
    ablation.add_argument("--output-root", default="result")
    ablation.add_argument("--disable-tqdm", action="store_true")
    ablation.add_argument("--print-components", action="store_true")
    ablation.set_defaults(func=_cmd_ablation)

    audit = subparsers.add_parser("audit", help="Audit a full-pipeline experiment run.")
    audit.add_argument("--run-root", required=True)
    audit.add_argument("--training-log", default=None)
    audit.add_argument("--predictions", default=None)
    audit.add_argument("--metrics", default=None)
    audit.add_argument("--require-nonempty-metrics", action="store_true")
    audit.add_argument("--allow-empty-split", action="store_true")
    audit.add_argument("--write-report", action="store_true")
    audit.add_argument("--report-path", default=None)
    audit.add_argument("--strict", action="store_true")
    audit.add_argument("--sample-limit", type=int, default=5)
    audit.set_defaults(func=_cmd_audit)

    suite = subparsers.add_parser("suite", help="Plan and run a reproducible experiment suite.")
    suite.add_argument("--suite", required=True)
    suite.add_argument("--config", default=DEFAULT_CONFIG)
    suite.add_argument("--dataset", nargs="+", default=None)
    suite.add_argument("--seed", nargs="+", type=int, default=None)
    suite.add_argument("--epochs", type=int, default=None)
    suite.add_argument("--limit", type=int, default=None)
    suite.add_argument("--device", default="cpu")
    suite.add_argument("--output-root", default="result")
    suite.add_argument("--split-file", default=None)
    suite.add_argument("--dry-run", action="store_true")
    suite.add_argument("--resume", action="store_true")
    suite.add_argument("--skip-complete", action="store_true")
    suite.add_argument("--audit-after-run", action="store_true")
    suite.add_argument("--strict", action="store_true")
    suite.add_argument("--require-nonempty-metrics", action="store_true")
    suite.add_argument("--disable-tqdm", action="store_true")
    suite.add_argument("--print-components", action="store_true")
    suite.add_argument("--batch-size", type=int, default=16)
    suite.add_argument("--lr", type=float, default=1e-4)
    suite.add_argument("--patience", type=int, default=3)
    suite.add_argument("--min-delta", type=float, default=0.0)
    suite.add_argument("--early-stop-metric", default="val_macro_f1")
    suite.set_defaults(func=_cmd_suite)

    preflight = subparsers.add_parser("preflight", help="Run Experiment 0 readiness preflight.")
    preflight.add_argument("--profile", required=True, choices=["smoke", "main_experiment"])
    preflight.add_argument("--config", default=DEFAULT_CONFIG)
    preflight.add_argument("--dataset", nargs="+", default=["harm_c", "harm_p", "facebook", "memotion"])
    preflight.add_argument("--seed", nargs="+", type=int, default=[42])
    preflight.add_argument("--label-set", choices=["full", "clean"], default="clean")
    preflight.add_argument("--normalized-root", default="dataset/annotation_normalized")
    preflight.add_argument("--vocab-path", default="configs/label_vocab.yaml")
    preflight.add_argument("--device", default="cpu")
    preflight.add_argument("--output-root", default="result")
    preflight.add_argument("--write-report", action="store_true")
    preflight.add_argument("--strict", action="store_true")
    preflight.add_argument("--fail-on-warnings", action="store_true")
    preflight.add_argument("--create-missing-splits", dest="create_missing_splits", action="store_true", default=None)
    preflight.add_argument("--no-create-missing-splits", dest="create_missing_splits", action="store_false")
    preflight.add_argument("--overwrite-splits", action="store_true")
    preflight.add_argument("--probe-pipeline", action="store_true")
    preflight.add_argument("--probe-limit", type=int, default=3)
    preflight.add_argument("--allow-fallback", action="store_true")
    preflight.add_argument("--allow-download", action="store_true")
    preflight.set_defaults(func=_cmd_preflight)
    return parser


def _add_ours_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", nargs="+", default=["harm_c"])
    parser.add_argument("--experiment", default="ours_full")
    parser.add_argument("--seed", nargs="+", type=int, default=[42])
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--early-stop-metric", default="val_macro_f1")
    parser.add_argument("--early-stop-mode", choices=["max", "min"], default="max")
    parser.add_argument("--save-best", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-last", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--print-components", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--output-root", default="result")
    parser.add_argument("--unfreeze-backbones", action="store_true")
    parser.add_argument("--no-relevance-mlp-training", action="store_true")
    parser.add_argument("--harmfulness-only", action="store_true")
    parser.add_argument("--label-set", choices=["full", "clean"], default="full")
    parser.add_argument("--normalized-root", default="dataset/annotation_normalized")
    parser.add_argument("--vocab-path", default="configs/label_vocab.yaml")
    parser.add_argument("--use-normalized-labels", dest="use_normalized_labels", action="store_true", default=True)
    parser.add_argument("--no-normalized-labels", dest="use_normalized_labels", action="store_false")
    parser.add_argument("--require-normalized-label", dest="require_normalized_label", action="store_true", default=True)
    parser.add_argument("--allow-missing-normalized-label", dest="require_normalized_label", action="store_false")
    parser.add_argument("--use-sample-weight", dest="use_sample_weight", action="store_true", default=True)
    parser.add_argument("--no-sample-weight", dest="use_sample_weight", action="store_false")


def _add_baseline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", nargs="+", default=["harm_c"])
    parser.add_argument("--baseline", choices=["image_only_clip", "text_only_encoder", "clip_text_concat"], default="text_only_encoder")
    parser.add_argument("--seed", nargs="+", type=int, default=[42])
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--early-stop-metric", default="val_macro_f1")
    parser.add_argument("--early-stop-mode", choices=["max", "min"], default="max")
    parser.add_argument("--save-best", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-last", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--output-root", default="result")
    parser.add_argument("--label-set", choices=["full", "clean"], default="full")
    parser.add_argument("--normalized-root", default="dataset/annotation_normalized")
    parser.add_argument("--vocab-path", default="configs/label_vocab.yaml")
    parser.add_argument("--use-normalized-labels", action="store_true", default=False)
    parser.add_argument("--require-normalized-label", dest="require_normalized_label", action="store_true", default=True)
    parser.add_argument("--allow-missing-normalized-label", dest="require_normalized_label", action="store_false")
    parser.add_argument("--use-sample-weight", action="store_true", default=False)


def _cmd_train(args: argparse.Namespace) -> None:
    datasets = normalize_dataset_names(args.dataset) or ["harm_c", "harm_p", "facebook", "memotion"]
    seeds = DEFAULT_SEEDS if args.all_seeds else args.seed
    for dataset in datasets:
        for seed in seeds:
            cfg = OursRunConfig(
                dataset_name=dataset,
                seed=seed,
                config_path=args.config,
                split_file=args.split_file,
                output_root=args.output_root,
                epochs=args.epochs,
                lr=args.lr,
                patience=args.patience,
                min_delta=args.min_delta,
                early_stop_metric=args.early_stop_metric,
                early_stop_mode=args.early_stop_mode,
                save_best=args.save_best,
                save_last=args.save_last,
                disable_tqdm=args.disable_tqdm,
                print_components=args.print_components,
                device=args.device,
                limit=args.limit,
                freeze_backbones=not args.unfreeze_backbones,
                train_relevance_mlp=not args.no_relevance_mlp_training,
                harmfulness_only=args.harmfulness_only,
                structured_auxiliary=not args.harmfulness_only,
                normalized_root=args.normalized_root,
                label_set=args.label_set,
                vocab_path=args.vocab_path,
                use_normalized_labels=args.use_normalized_labels,
                require_normalized_label=args.require_normalized_label,
                use_sample_weight=args.use_sample_weight,
            )
            metrics = run_ours_experiment(cfg)
            print(f"{dataset}/{args.experiment}/seed={seed}: macro_f1={metrics.get('macro_f1')} accuracy={metrics.get('accuracy')}")


def _cmd_baseline(args: argparse.Namespace) -> None:
    datasets = normalize_dataset_names(args.dataset) or ["harm_c", "harm_p", "facebook", "memotion"]
    seeds = DEFAULT_SEEDS if args.all_seeds else args.seed
    for dataset in datasets:
        for seed in seeds:
            cfg = BaselineRunConfig(
                model_name=args.baseline,
                dataset_name=dataset,
                seed=seed,
                config_path=args.config,
                split_file=args.split_file,
                output_root=args.output_root,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                patience=args.patience,
                min_delta=args.min_delta,
                early_stop_metric=args.early_stop_metric,
                early_stop_mode=args.early_stop_mode,
                save_best=args.save_best,
                save_last=args.save_last,
                disable_tqdm=args.disable_tqdm,
                device=args.device,
                limit=args.limit,
                normalized_root=args.normalized_root,
                label_set=args.label_set,
                vocab_path=args.vocab_path,
                use_normalized_labels=args.use_normalized_labels,
                require_normalized_label=args.require_normalized_label,
                use_sample_weight=args.use_sample_weight,
            )
            metrics = run_baseline_experiment(cfg)
            print(f"{dataset}/{args.baseline}/seed={seed}: macro_f1={metrics.get('macro_f1')} accuracy={metrics.get('accuracy')}")


def _cmd_stage(args: argparse.Namespace) -> None:
    overrides: dict[str, Any] = {}
    if args.device:
        overrides = {"runtime": {"device": args.device}}
    runner = PipelineRunner(args.config, overrides=overrides)
    until = args.until.lower().replace("stage_", "")
    runner.run(dataset_names=args.dataset, limit=args.limit, run_until=until, save=not args.no_save)


def _cmd_evaluate(args: argparse.Namespace) -> None:
    prediction_path = (
        Path(args.prediction_file)
        if args.prediction_file
        else Path(args.result_root) / "predictions" / args.dataset / args.model / str(args.seed) / "final_predictions.jsonl"
    )
    metrics = evaluate_prediction_file(
        prediction_path,
        dataset=args.dataset,
        output_root=args.output_root,
        disable_tqdm=args.disable_tqdm,
    )
    print(f"{args.dataset}/{args.model}/seed={args.seed}: structured metrics saved to {args.output_root}")
    print(metrics)


def _cmd_ablation(args: argparse.Namespace) -> None:
    datasets = normalize_dataset_names(args.dataset) or ["harm_c", "harm_p", "facebook", "memotion"]
    seeds = DEFAULT_SEEDS if args.all_seeds else args.seed
    ablations = _resolve_ablations(args.experiment, args.ablation)
    fusion_modes = [] if not args.fusion_mode else args.fusion_mode
    if "all" in fusion_modes:
        from experiments.ablation_configs import FUSION_MODES

        fusion_modes = FUSION_MODES
    for dataset in datasets:
        for seed in seeds:
            for ablation in ablations:
                metrics = run_ablation_experiment(
                    dataset,
                    ablation,
                    seed=seed,
                    config_path=args.config,
                    split_file=args.split_file,
                    output_root=args.output_root,
                    limit=args.limit,
                    disable_tqdm=args.disable_tqdm,
                    print_components=args.print_components,
                )
                print(f"{dataset}/ablation_{ablation}/seed={seed}: macro_f1={metrics.get('macro_f1')}")
            for mode in fusion_modes:
                metrics = run_fusion_experiment(
                    dataset,
                    mode,
                    seed=seed,
                    config_path=args.config,
                    split_file=args.split_file,
                    output_root=args.output_root,
                    limit=args.limit,
                    disable_tqdm=args.disable_tqdm,
                    print_components=args.print_components,
                )
                print(f"{dataset}/fusion_{mode}/seed={seed}: macro_f1={metrics.get('macro_f1')}")


def _cmd_audit(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root)
    result = audit_run_artifacts(
        run_root,
        training_log=args.training_log,
        predictions=args.predictions,
        metrics=args.metrics,
        require_nonempty_metrics=args.require_nonempty_metrics,
        allow_empty_split=args.allow_empty_split,
        strict=args.strict,
        sample_limit=args.sample_limit,
    )
    if args.write_report:
        report_path = Path(args.report_path) if args.report_path else run_root / "pipeline_audit_report.md"
        written = write_audit_report(result, report_path)
        print(f"Report: {written}")
    print(format_audit_summary(result))
    if result["errors"]:
        raise SystemExit(1)


def _cmd_suite(args: argparse.Namespace) -> None:
    result = run_suite(args)
    if not args.dry_run and result.get("status") != "complete":
        raise SystemExit(1)


def _cmd_preflight(args: argparse.Namespace) -> None:
    result = run_preflight(
        profile=args.profile,
        config_path=args.config,
        datasets=args.dataset,
        seeds=args.seed,
        label_set=args.label_set,
        normalized_root=args.normalized_root,
        vocab_path=args.vocab_path,
        device=args.device,
        output_root=args.output_root,
        strict=args.strict,
        fail_on_warnings=args.fail_on_warnings,
        create_missing_splits=args.create_missing_splits,
        overwrite_splits=args.overwrite_splits,
        probe_pipeline=args.probe_pipeline,
        probe_limit=args.probe_limit,
        allow_fallback=args.allow_fallback,
        allow_download=args.allow_download,
        write_report=args.write_report,
    )
    print(format_preflight_summary(result))
    if args.strict and result.errors:
        raise SystemExit(1)
    if args.fail_on_warnings and result.warnings:
        raise SystemExit(1)


def _resolve_ablations(experiment: str | None, requested: list[str] | None) -> list[str]:
    if requested:
        return ABLATION_MODES if "all" in requested else requested
    if experiment:
        name = experiment.split(".")[-1]
        aliases = {
            "no_retrieval": "w_o_retrieval",
            "no_verifier": "w_o_support_verifier",
            "w_o_verifier": "w_o_support_verifier",
            "no_task_aware_gate": "w_o_task_aware_gate",
        }
        return [aliases.get(name, name)]
    return ["full"]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
