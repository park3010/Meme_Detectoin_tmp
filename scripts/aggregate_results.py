"""Aggregate baseline metrics into CSV tables."""

from __future__ import annotations

import argparse

from common import ROOT  # noqa: F401 - ensures repo root is importable when run as a script

from experiments.aggregate_results import write_aggregate_tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-root", default="result/predictions")
    parser.add_argument("--output-root", default="result/metrics")
    args = parser.parse_args()
    main_path, mean_std_path = write_aggregate_tables(args.predictions_root, args.output_root)
    print(f"Saved per-seed metrics: {main_path}")
    print(f"Saved mean/std metrics: {mean_std_path}")


if __name__ == "__main__":
    main()
