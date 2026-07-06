"""Analysis subcommands for the unified CLI."""

from __future__ import annotations

import argparse

from experiments.ablation_configs import KNOWLEDGE_MODES
from experiments.cross_domain import CrossDomainConfig, run_cross_domain
from experiments.error_case_analysis import select_error_cases
from experiments.knowledge_comparison import run_knowledge_comparison
from experiments.progress import progress_config_from_flags
from experiments.rationale_eval import run_rationale_evaluation
from experiments.runtime_cost import run_runtime_cost_analysis
from experiments.significance import run_significance_tests
from experiments.splits import DEFAULT_SEEDS, normalize_dataset_names
from experiments.subset_analysis import run_subset_analysis
from experiments.verifier_eval import run_verifier_evaluation


def add_analysis_commands(subparsers: argparse._SubParsersAction, default_config: str) -> None:
    analysis = subparsers.add_parser("analysis", help="Paper-support analysis and diagnostic runners.")
    analysis_subparsers = analysis.add_subparsers(dest="analysis_command", required=True)

    knowledge = analysis_subparsers.add_parser("knowledge-comparison", help="Run evaluation-time knowledge modes.")
    knowledge.add_argument("--config", default=default_config)
    knowledge.add_argument("--dataset", nargs="+", default=["harm_c"])
    knowledge.add_argument("--mode", nargs="+", default=["verified"])
    knowledge.add_argument("--seed", nargs="+", type=int, default=[42])
    knowledge.add_argument("--all-seeds", action="store_true")
    knowledge.add_argument("--limit", type=int, default=None)
    knowledge.add_argument("--split-file", default=None)
    knowledge.add_argument("--output-root", default="result")
    knowledge.add_argument("--device", default="cpu")
    knowledge.add_argument("--print-components", action="store_true")
    _add_progress_args(knowledge)
    knowledge.set_defaults(func=_cmd_knowledge_comparison)

    cross = analysis_subparsers.add_parser("cross-domain", help="Run cross-dataset robustness.")
    cross.add_argument("--setting", required=True, choices=["in_domain", "train_one_test_others", "mixed_train", "leave_one_domain_out"])
    cross.add_argument("--model", default="ours_full")
    cross.add_argument("--seed", type=int, default=42)
    cross.add_argument("--config", default=default_config)
    cross.add_argument("--train-dataset", default=None)
    cross.add_argument("--heldout", default=None)
    cross.add_argument("--test-dataset", default=None)
    cross.add_argument("--epochs", type=int, default=0)
    cross.add_argument("--lr", type=float, default=1e-4)
    cross.add_argument("--device", default="cpu")
    cross.add_argument("--limit", type=int, default=None)
    cross.add_argument("--output-root", default="result")
    _add_progress_args(cross)
    cross.set_defaults(func=_cmd_cross_domain)

    verifier = analysis_subparsers.add_parser("verifier", help="Evaluate Stage C verifier behavior.")
    verifier.add_argument("--dataset", nargs="+", default=["harm_c"])
    verifier.add_argument("--seed", type=int, default=42)
    verifier.add_argument("--config", default=default_config)
    verifier.add_argument("--limit", type=int, default=None)
    verifier.add_argument("--output-root", default="result")
    _add_progress_args(verifier)
    verifier.set_defaults(func=_cmd_verifier)

    subset = analysis_subparsers.add_parser("subset", help="Run difficult-subset analysis.")
    subset.add_argument("--dataset", default="all")
    subset.add_argument("--model", default="ours_full")
    subset.add_argument("--seed", type=int, default=42)
    subset.add_argument("--result-root", default="result")
    subset.set_defaults(func=_cmd_subset)

    rationale = analysis_subparsers.add_parser("rationale", help="Run rationale quality proxy evaluation.")
    rationale.add_argument("--dataset", default="all")
    rationale.add_argument("--model", default="ours_full")
    rationale.add_argument("--seed", type=int, default=42)
    rationale.add_argument("--result-root", default="result")
    _add_progress_args(rationale)
    rationale.set_defaults(func=_cmd_rationale)

    runtime = analysis_subparsers.add_parser("runtime", help="Measure runtime/cost metrics.")
    runtime.add_argument("--dataset", default="harm_c")
    runtime.add_argument("--limit", type=int, default=20)
    runtime.add_argument("--device", default="cpu")
    runtime.add_argument("--warmup", type=int, default=1)
    runtime.add_argument("--config", default=default_config)
    runtime.add_argument("--output-root", default="result")
    runtime.add_argument("--print-components", action="store_true")
    _add_progress_args(runtime)
    runtime.set_defaults(func=_cmd_runtime)

    significance = analysis_subparsers.add_parser("significance", help="Run multi-seed significance tests.")
    significance.add_argument("--result-root", default="result")
    significance.add_argument("--metric", default="macro_f1")
    significance.add_argument("--output", default="result/metrics/significance_tests.csv")
    significance.set_defaults(func=_cmd_significance)

    cases = analysis_subparsers.add_parser("select-error-cases", help="Select TP/TN/FP/FN case-study rows.")
    cases.add_argument("--dataset", default="all")
    cases.add_argument("--model", default="ours_full")
    cases.add_argument("--seed", type=int, default=42)
    cases.add_argument("--result-root", default="result")
    _add_progress_args(cases)
    cases.set_defaults(func=_cmd_select_error_cases)


def _cmd_knowledge_comparison(args: argparse.Namespace) -> None:
    datasets = normalize_dataset_names(args.dataset) or ["harm_c", "harm_p", "facebook", "memotion"]
    modes = KNOWLEDGE_MODES if "all" in args.mode else args.mode
    seeds = DEFAULT_SEEDS if args.all_seeds else args.seed
    progress = _progress_config(args)
    for dataset in datasets:
        for seed in seeds:
            for mode in modes:
                metrics = run_knowledge_comparison(
                    dataset,
                    mode,
                    seed=seed,
                    config_path=args.config,
                    split_file=args.split_file,
                    output_root=args.output_root,
                    limit=args.limit,
                    disable_tqdm=args.disable_tqdm,
                    progress=progress,
                    print_components=args.print_components,
                    device=args.device,
                )
                print(f"{dataset}/knowledge_{mode}/seed={seed}: macro_f1={metrics.get('macro_f1')}")


def _cmd_cross_domain(args: argparse.Namespace) -> None:
    rows = run_cross_domain(
        CrossDomainConfig(
            setting=args.setting,
            model_name=args.model,
            seed=args.seed,
            config_path=args.config,
            train_dataset=args.train_dataset,
            heldout=args.heldout,
            test_dataset=args.test_dataset,
            epochs=args.epochs,
            lr=args.lr,
            device=args.device,
            limit=args.limit,
            output_root=args.output_root,
            disable_tqdm=args.disable_tqdm,
            progress=_progress_config(args),
        )
    )
    print(f"Wrote {len(rows)} cross-domain row(s).")


def _cmd_verifier(args: argparse.Namespace) -> None:
    progress = _progress_config(args)
    datasets = normalize_dataset_names(args.dataset) or ["harm_c", "harm_p", "facebook", "memotion"]
    for dataset in datasets:
        metrics = run_verifier_evaluation(
            dataset,
            seed=args.seed,
            config_path=args.config,
            output_root=args.output_root,
            limit=args.limit,
            disable_tqdm=args.disable_tqdm,
            progress=progress,
        )
        print(f"{dataset}: relevance_f1={metrics.get('relevance_f1')} support_macro_f1={metrics.get('support_macro_f1')}")


def _cmd_subset(args: argparse.Namespace) -> None:
    rows = run_subset_analysis(args.dataset, args.model, args.seed, args.result_root)
    print(f"Wrote {len(rows)} subset rows.")


def _cmd_rationale(args: argparse.Namespace) -> None:
    rows = run_rationale_evaluation(
        args.dataset,
        args.model,
        args.seed,
        args.result_root,
        disable_tqdm=args.disable_tqdm,
        progress=_progress_config(args),
    )
    print(f"Wrote {len(rows)} rationale rows.")


def _cmd_runtime(args: argparse.Namespace) -> None:
    summary = run_runtime_cost_analysis(
        args.dataset,
        args.limit,
        args.device,
        args.warmup,
        args.config,
        args.output_root,
        disable_tqdm=args.disable_tqdm,
        progress=_progress_config(args),
        print_components=args.print_components,
    )
    print(summary)


def _cmd_significance(args: argparse.Namespace) -> None:
    rows = run_significance_tests(args.result_root, args.metric, args.output)
    print(f"Wrote {len(rows)} significance rows to {args.output}")


def _cmd_select_error_cases(args: argparse.Namespace) -> None:
    cases = select_error_cases(
        args.dataset,
        args.model,
        args.seed,
        args.result_root,
        disable_tqdm=args.disable_tqdm,
        progress=_progress_config(args),
    )
    print(f"Selected {len(cases)} cases.")


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


__all__ = ["add_analysis_commands"]
