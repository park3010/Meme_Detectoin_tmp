"""Report/export subcommands for the unified CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import print_json
from experiments.posthoc_error_analysis import export_case_visualization_data
from experiments.evaluation import evaluate_prediction_file, write_structured_aggregate_tables
from experiments.progress import progress_config_from_flags
from experiments.reporting import export_intermediate_manifest, export_paper_tables, write_aggregate_tables


def add_report_commands(subparsers: argparse._SubParsersAction, _default_config: str) -> None:
    report = subparsers.add_parser("report", help="Aggregate metrics and export analysis tables.")
    report_subparsers = report.add_subparsers(dest="report_command", required=True)

    aggregate = report_subparsers.add_parser("aggregate", help="Aggregate harmfulness metrics.")
    aggregate.add_argument("--predictions-root", default="result/predictions")
    aggregate.add_argument("--output-root", default="result/metrics")
    aggregate.set_defaults(func=_cmd_aggregate)

    structured = report_subparsers.add_parser("aggregate-structured", help="Aggregate structured metrics.")
    structured.add_argument("--predictions-root", default="result/predictions")
    structured.add_argument("--output-root", default="result/metrics")
    _add_progress_args(structured)
    structured.set_defaults(func=_cmd_aggregate_structured)

    evaluate = report_subparsers.add_parser("evaluate-structured", help="Evaluate one structured prediction run.")
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--model", default="ours_full")
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--prediction-file", default=None)
    evaluate.add_argument("--result-root", default="result")
    evaluate.add_argument("--output-root", default="result/metrics")
    _add_progress_args(evaluate)
    evaluate.set_defaults(func=_cmd_evaluate_structured)

    paper = report_subparsers.add_parser("export-paper-tables", help="Export paper-ready CSV tables.")
    paper.add_argument("--result-root", default="result")
    paper.add_argument("--output-root", default=None)
    paper.set_defaults(func=_cmd_export_paper_tables)

    cases = report_subparsers.add_parser("export-case-data", help="Export case-study visualization JSONL.")
    cases.add_argument("--dataset", default="all")
    cases.add_argument("--model", default="ours_full")
    cases.add_argument("--seed", type=int, default=42)
    cases.add_argument("--result-root", default="result")
    _add_progress_args(cases)
    cases.set_defaults(func=_cmd_export_case_data)

    intermediate = report_subparsers.add_parser("export-intermediate", help="Export a manifest of intermediate result files.")
    intermediate.add_argument("--result-root", default="result")
    intermediate.add_argument("--output", default="result/intermediate_manifest.json")
    intermediate.set_defaults(func=_cmd_export_intermediate)


def _cmd_aggregate(args: argparse.Namespace) -> None:
    main_path, mean_std_path = write_aggregate_tables(args.predictions_root, args.output_root)
    print(f"Saved per-seed metrics: {main_path}")
    print(f"Saved mean/std metrics: {mean_std_path}")


def _cmd_aggregate_structured(args: argparse.Namespace) -> None:
    progress = _progress_config(args)
    per_run, mean_std = write_structured_aggregate_tables(
        args.predictions_root,
        args.output_root,
        disable_tqdm=args.disable_tqdm,
        progress=progress,
    )
    print(f"Saved structured metrics: {per_run}")
    print(f"Saved structured mean/std: {mean_std}")


def _cmd_evaluate_structured(args: argparse.Namespace) -> None:
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
        progress=_progress_config(args),
    )
    print(f"{args.dataset}/{args.model}/seed={args.seed}: structured metrics saved to {args.output_root}")
    print(metrics)


def _cmd_export_paper_tables(args: argparse.Namespace) -> None:
    for path in export_paper_tables(args.result_root, args.output_root):
        print(f"Wrote {path}")


def _cmd_export_case_data(args: argparse.Namespace) -> None:
    path = export_case_visualization_data(
        dataset=args.dataset,
        model=args.model,
        seed=args.seed,
        result_root=args.result_root,
        disable_tqdm=args.disable_tqdm,
        progress=_progress_config(args),
    )
    print(f"Wrote {path}")


def _cmd_export_intermediate(args: argparse.Namespace) -> None:
    print_json(export_intermediate_manifest(args.result_root, args.output))


def _add_progress_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--tqdm-mininterval", type=float, default=0.5)
    parser.add_argument("--tqdm-leave", action="store_true")


def _progress_config(args: argparse.Namespace):
    return progress_config_from_flags(
        disable_tqdm=bool(getattr(args, "disable_tqdm", False)),
        tqdm_mininterval=float(getattr(args, "tqdm_mininterval", 0.5)),
        tqdm_leave=bool(getattr(args, "tqdm_leave", False)),
    )


__all__ = ["add_report_commands"]
