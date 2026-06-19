"""Audit a full-framework run for provenance and experiment readiness."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import ROOT  # noqa: F401 - importing common also adds the repository root to sys.path.

from experiments.pipeline_audit import audit_run_artifacts, format_audit_summary, write_audit_report


def build_parser() -> argparse.ArgumentParser:
    """Build the pipeline artifact audit CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--training-log", default=None)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--require-nonempty-metrics", action="store_true")
    parser.add_argument("--allow-empty-split", action="store_true")
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--sample-limit", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
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


if __name__ == "__main__":
    main()
