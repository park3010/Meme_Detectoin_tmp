"""Unified command-line entry point for framework training and evaluation."""

from __future__ import annotations

from common import ROOT  # noqa: F401 - importing common also adds the repository root to sys.path.

from commands.experiment import (
    BaselineRunConfig,
    OursRunConfig,
    build_parser,
    run_baseline_experiment,
    run_ours_experiment,
    run_suite,
)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


__all__ = [
    "BaselineRunConfig",
    "OursRunConfig",
    "build_parser",
    "main",
    "run_baseline_experiment",
    "run_ours_experiment",
    "run_suite",
]
