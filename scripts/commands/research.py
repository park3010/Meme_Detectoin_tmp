"""Locked HarMeme-to-FHM research protocol CLI commands."""

from __future__ import annotations

import argparse

from common import print_json
from experiments.research_orchestration import (
    audit_research_runs,
    execute_research_suite,
    plan_research,
    research_status,
    run_research_preflight,
)
from experiments.research_dashboard import build_research_dashboard
from experiments.research_results import aggregate_research_results
from experiments.smoke_diagnostics import diagnose_smoke
from experiments.source_sanity import run_source_sanity
from experiments.duplicate_adjudication import run_duplicate_adjudication
from experiments.research_error_analysis import export_research_error_cases
from experiments.research_human_eval import agreement_report, export_human_evaluation, import_human_ratings, validate_human_ratings
from experiments.paper_export import check_paper, export_research_paper_artifacts
from experiments.retrieval_corpus import (
    CANONICAL_PROFILE,
    DEFAULT_OUTPUT_ROOT,
    audit_retrieval_profile,
    build_retrieval_corpus,
    retrieval_status,
)


def add_research_commands(subparsers: argparse._SubParsersAction, default_config: str) -> None:
    research = subparsers.add_parser("research", help="Plan and run the locked HarMeme-to-FHM paper protocol.")
    commands = research.add_subparsers(dest="research_command", required=True)

    plan = commands.add_parser("plan", help="Resolve registry and suite status without training.")
    plan.add_argument("--suite", default=None)
    _common_paths(plan, default_config)
    plan.set_defaults(func=_cmd_plan)

    preflight = commands.add_parser("preflight", help="Create manifests and run protocol/leakage checks.")
    _common_paths(preflight, default_config)
    preflight.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    preflight.add_argument("--force-regenerate-split", action="store_true")
    preflight.set_defaults(func=_cmd_preflight)

    retrieval_build = commands.add_parser(
        "retrieval-build", help="Build the immutable HarMeme-train retrieval corpus offline."
    )
    retrieval_build.add_argument("--registry", default="configs/experiment_registry.yaml")
    retrieval_build.add_argument("--config", default=default_config)
    retrieval_build.add_argument("--profile", default=CANONICAL_PROFILE)
    retrieval_build.add_argument("--source-partition", default="train")
    retrieval_build.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    retrieval_build.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True)
    retrieval_build.add_argument("--force", action="store_true")
    retrieval_build.add_argument("--limit", type=int, default=None, help="Test/smoke builds only.")
    retrieval_build.set_defaults(func=_cmd_retrieval_build)

    retrieval_audit = commands.add_parser(
        "retrieval-audit", help="Verify retrieval provenance, corpus, and index hashes."
    )
    retrieval_audit.add_argument("--registry", default="configs/experiment_registry.yaml")
    retrieval_audit.add_argument("--config", default=default_config)
    retrieval_audit.add_argument("--profile", default=CANONICAL_PROFILE)
    retrieval_audit.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    retrieval_audit.set_defaults(func=_cmd_retrieval_audit)

    retrieval_status_parser = commands.add_parser(
        "retrieval-status", help="Show the configured retrieval profile state."
    )
    retrieval_status_parser.add_argument("--registry", default="configs/experiment_registry.yaml")
    retrieval_status_parser.add_argument("--config", default=default_config)
    retrieval_status_parser.add_argument("--profile", default=CANONICAL_PROFILE)
    retrieval_status_parser.set_defaults(func=_cmd_retrieval_status)

    for name in ("run", "resume"):
        runner = commands.add_parser(name, help=f"{name.title()} a registered suite; planning is the default.")
        runner.add_argument("--suite", required=True)
        runner.add_argument(
            "--experiment",
            nargs="+",
            default=None,
            help="Run only these experiment IDs from the selected suite.",
        )
        _common_paths(runner, default_config)
        runner.add_argument("--execute", action="store_true", help="Compatibility flag; execution is the default unless --dry-run is set.")
        runner.add_argument("--dry-run", action="store_true", help="Resolve and report runs without training or inference.")
        if name == "run":
            runner.add_argument("--resume", action="store_true", help="Skip audited-complete runs and continue partial suites.")
        runner.add_argument("--force", action="store_true")
        runner.add_argument("--epochs", type=int, default=None)
        runner.add_argument("--limit", type=int, default=None)
        runner.add_argument("--device", default="cpu")
        runner.add_argument("--disable-tqdm", action="store_true")
        runner.set_defaults(func=_cmd_run, resume=name == "resume")

    status = commands.add_parser("status", help="Summarize canonical research run artifacts.")
    status.add_argument("--suite", default=None)
    status.add_argument("--output-root", default="result")
    status.set_defaults(func=_cmd_status)

    audit = commands.add_parser("audit", help="Audit canonical runs and global FHM leakage state.")
    audit.add_argument("--suite", default=None)
    audit.add_argument("--output-root", default="result")
    audit.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    audit.set_defaults(func=_cmd_audit)

    diagnose = commands.add_parser("diagnose-smoke", help="Forensically diagnose immutable smoke artifacts without training.")
    diagnose.add_argument("--suite", required=True)
    diagnose.add_argument("--output-root", default="result")
    diagnose.add_argument("--reference-experiment", default="ours_full")
    diagnose.add_argument("--strict", action=argparse.BooleanOptionalAction, default=False)
    diagnose.add_argument("--write-report", action="store_true")
    diagnose.add_argument("--force", action="store_true")
    diagnose.set_defaults(func=_cmd_diagnose_smoke)

    source_sanity = commands.add_parser("source-sanity", help="Run source-only HarMeme sanity diagnostics; FHM/Memotion are forbidden.")
    source_sanity.add_argument("--profile", required=True, choices=["data_integrity","gradient_check","overfit_32","overfit_128","shuffled_label","domain_probe","all"])
    source_sanity.add_argument("--output-root", default="result/source_sanity")
    source_sanity.add_argument("--config", default=default_config)
    source_sanity.add_argument("--seed", type=int, default=42)
    source_sanity.add_argument("--device", default="cpu")
    source_sanity.add_argument("--strict", action=argparse.BooleanOptionalAction, default=False)
    source_sanity.add_argument("--force", action="store_true")
    source_sanity.add_argument("--disable-tqdm", action="store_true")
    source_sanity.set_defaults(func=_cmd_source_sanity)

    adjudication = commands.add_parser("duplicate-adjudication", help="Read-only HarMeme duplicate-image adjudication and split-v2 preview.")
    adjudication.add_argument("--output-root", default="result/source_sanity")
    adjudication.add_argument("--strict", action=argparse.BooleanOptionalAction, default=False)
    adjudication.add_argument("--write-report", action="store_true")
    adjudication.add_argument("--force", action="store_true")
    adjudication.set_defaults(func=_cmd_duplicate_adjudication)

    aggregate = commands.add_parser("aggregate", help="Build canonical long-form and seed-summary results.")
    aggregate.add_argument("--registry", default="configs/experiment_registry.yaml")
    aggregate.add_argument("--output-root", default="result")
    aggregate.add_argument("--results-root", default=None, help="Alias for --output-root used by isolated research runs.")
    aggregate.set_defaults(func=_cmd_aggregate)

    dashboard = commands.add_parser("dashboard", help="Build the static offline research dashboard.")
    dashboard.add_argument("--output-root", default="result")
    dashboard.add_argument("--results-root", default=None, help="Alias for --output-root used by isolated research runs.")
    dashboard.set_defaults(func=_cmd_dashboard)

    human_export = commands.add_parser("human-export", help="Export blinded, unrated evidence/rationale forms.")
    human_export.add_argument("--suite", default="harmeme_to_fhm_1seed")
    human_export.add_argument("--experiment", default="ours_full")
    human_export.add_argument("--seed", type=int, default=42)
    human_export.add_argument("--limit", type=int, default=100)
    human_export.add_argument("--random-seed", type=int, default=42)
    human_export.add_argument("--output-root", default="result")
    human_export.add_argument("--human-eval-root", default="human_eval")
    human_export.set_defaults(func=_cmd_human_export)

    human_validate = commands.add_parser("human-validate", help="Validate a completed human-rating CSV.")
    human_validate.add_argument("--input", required=True)
    human_validate.add_argument("--schema", required=True)
    human_validate.set_defaults(func=_cmd_human_validate)

    human_import = commands.add_parser("human-import", help="Validate and import supplied human ratings.")
    human_import.add_argument("--input", required=True)
    human_import.add_argument("--schema", required=True)
    human_import.add_argument("--human-eval-root", default="human_eval")
    human_import.set_defaults(func=_cmd_human_import)

    human_agreement = commands.add_parser("human-agreement", help="Compute agreement for validated ratings.")
    human_agreement.add_argument("--input", required=True)
    human_agreement.add_argument("--rating-column", nargs="+", required=True)
    human_agreement.set_defaults(func=_cmd_human_agreement)

    errors = commands.add_parser("error-export", help="Export FHM TP/TN/FP/FN analysis packages.")
    errors.add_argument("--suite", default="harmeme_to_fhm_1seed")
    errors.add_argument("--experiment", default="ours_full")
    errors.add_argument("--seed", type=int, default=42)
    errors.add_argument("--per-category", type=int, default=10)
    errors.add_argument("--output-root", default="result")
    errors.set_defaults(func=_cmd_error_export)

    paper_export = commands.add_parser("paper-export", help="Regenerate generated-only LaTeX inputs.")
    paper_export.add_argument("--output-root", default="result")
    paper_export.add_argument("--latex-root", default="latex")
    paper_export.add_argument("--results-root", default=None)
    paper_export.add_argument("--paper-root", default=None)
    paper_export.set_defaults(func=_cmd_paper_export)

    paper_check = commands.add_parser("paper-check", help="Validate draft or final paper readiness.")
    paper_check.add_argument("--mode", choices=["draft", "final"], default="draft")
    paper_check.add_argument("--latex-root", default="latex")
    paper_check.add_argument("--paper-root", default=None)
    paper_check.add_argument("--results-root", default="result")
    paper_check.set_defaults(func=_cmd_paper_check)


def _common_paths(parser: argparse.ArgumentParser, default_config: str) -> None:
    parser.add_argument("--registry", default="configs/experiment_registry.yaml")
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--output-root", default="result")


def _cmd_plan(args: argparse.Namespace) -> None:
    print_json(plan_research(suite=args.suite, registry_path=args.registry, output_root=args.output_root))


def _cmd_preflight(args: argparse.Namespace) -> None:
    result = run_research_preflight(
        registry_path=args.registry,
        config_path=args.config,
        output_root=args.output_root,
        strict=args.strict,
        force_regenerate_split=args.force_regenerate_split,
    )
    print_json(result)
    if args.strict and not result["passed"]:
        raise SystemExit(2)


def _cmd_retrieval_build(args: argparse.Namespace) -> None:
    print_json(
        build_retrieval_corpus(
            registry_path=args.registry,
            config_path=args.config,
            profile=args.profile,
            source_partition=args.source_partition,
            output_root=args.output_root,
            offline=args.offline,
            force=args.force,
            limit=args.limit,
        )
    )


def _cmd_retrieval_audit(args: argparse.Namespace) -> None:
    result = audit_retrieval_profile(
        registry_path=args.registry,
        config_path=args.config,
        profile=args.profile,
        strict=args.strict,
    )
    print_json(result)
    if args.strict and not result["passed"]:
        raise SystemExit(2)


def _cmd_retrieval_status(args: argparse.Namespace) -> None:
    print_json(
        retrieval_status(
            registry_path=args.registry,
            config_path=args.config,
            profile=args.profile,
        )
    )


def _cmd_run(args: argparse.Namespace) -> None:
    result = execute_research_suite(
        args.suite,
        experiment_ids=args.experiment,
        registry_path=args.registry,
        config_path=args.config,
        output_root=args.output_root,
        execute=not args.dry_run,
        resume=args.resume,
        force=args.force,
        epochs=args.epochs,
        limit=args.limit,
        device=args.device,
        disable_tqdm=args.disable_tqdm,
    )
    print_json(result)


def _cmd_status(args: argparse.Namespace) -> None:
    print_json(research_status(output_root=args.output_root, suite=args.suite))


def _cmd_audit(args: argparse.Namespace) -> None:
    result = audit_research_runs(output_root=args.output_root, suite=args.suite, strict=args.strict)
    print_json(result)
    if args.strict and not result["passed"]:
        raise SystemExit(2)


def _cmd_diagnose_smoke(args: argparse.Namespace) -> None:
    try:
        result = diagnose_smoke(suite=args.suite, output_root=args.output_root, reference_experiment=args.reference_experiment, strict=args.strict, write_report=args.write_report, force=args.force)
    except RuntimeError as exc:
        print_json({"passed": False, "error": str(exc)})
        raise SystemExit(2) from exc
    print_json(result)


def _cmd_source_sanity(args: argparse.Namespace) -> None:
    try:
        result = run_source_sanity(profile=args.profile, output_root=args.output_root, config=args.config, seed=args.seed, device=args.device, strict=args.strict, force=args.force, disable_tqdm=args.disable_tqdm)
    except (RuntimeError, ValueError, FileExistsError) as exc:
        print_json({"passed": False, "error": str(exc), "source_only": True})
        raise SystemExit(2) from exc
    print_json(result)


def _cmd_duplicate_adjudication(args: argparse.Namespace) -> None:
    try:
        result=run_duplicate_adjudication(output_root=args.output_root,strict=args.strict,write_report=args.write_report,force=args.force)
    except (RuntimeError,ValueError,FileExistsError) as exc:
        print_json({"passed":False,"error":str(exc),"source_only":True})
        raise SystemExit(2) from exc
    print_json(result)


def _cmd_aggregate(args: argparse.Namespace) -> None:
    print_json(aggregate_research_results(output_root=args.results_root or args.output_root, registry_path=args.registry))


def _cmd_dashboard(args: argparse.Namespace) -> None:
    print(f"Wrote {build_research_dashboard(output_root=args.results_root or args.output_root)}")


def _cmd_human_export(args: argparse.Namespace) -> None:
    print_json(export_human_evaluation(output_root=args.output_root, suite=args.suite, experiment_id=args.experiment, seed=args.seed, limit=args.limit, random_seed=args.random_seed, human_eval_root=args.human_eval_root))


def _cmd_human_validate(args: argparse.Namespace) -> None:
    result = validate_human_ratings(args.input, args.schema)
    print_json(result)
    if not result["passed"]:
        raise SystemExit(2)


def _cmd_human_import(args: argparse.Namespace) -> None:
    result = import_human_ratings(args.input, args.schema, human_eval_root=args.human_eval_root)
    print_json(result)
    if not result["passed"]:
        raise SystemExit(2)


def _cmd_human_agreement(args: argparse.Namespace) -> None:
    print_json(agreement_report(args.input, args.rating_column))


def _cmd_error_export(args: argparse.Namespace) -> None:
    print_json(export_research_error_cases(output_root=args.output_root, suite=args.suite, experiment_id=args.experiment, seed=args.seed, per_category=args.per_category))


def _cmd_paper_export(args: argparse.Namespace) -> None:
    print_json(export_research_paper_artifacts(output_root=args.results_root or args.output_root, latex_root=args.paper_root or args.latex_root))


def _cmd_paper_check(args: argparse.Namespace) -> None:
    result = check_paper(mode=args.mode, latex_root=args.paper_root or args.latex_root, results_root=args.results_root)
    print_json(result)
    if not result["passed"]:
        raise SystemExit(2)


__all__ = ["add_research_commands"]
