"""Run multi-seed statistical significance tests."""

from __future__ import annotations

import argparse

from common import ROOT  # noqa: F401

from experiments.significance import run_significance_tests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--metric", default="macro_f1")
    parser.add_argument("--output", default="result/metrics/significance_tests.csv")
    args = parser.parse_args()
    rows = run_significance_tests(args.result_root, args.metric, args.output)
    print(f"Wrote {len(rows)} significance rows to {args.output}")


if __name__ == "__main__":
    main()
