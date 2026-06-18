"""Run difficult subset analysis on saved prediction files."""

from __future__ import annotations

import argparse

from common import ROOT  # noqa: F401

from experiments.subset_analysis import run_subset_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--model", default="ours_full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--result-root", default="result")
    args = parser.parse_args()
    rows = run_subset_analysis(args.dataset, args.model, args.seed, args.result_root)
    print(f"Wrote {len(rows)} subset rows.")


if __name__ == "__main__":
    main()
