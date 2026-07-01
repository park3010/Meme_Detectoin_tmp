"""Aggregate structured interpretation metrics across saved predictions."""

from __future__ import annotations

import argparse

from common import ROOT  # noqa: F401

from experiments.evaluation import write_structured_aggregate_tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-root", default="result/predictions")
    parser.add_argument("--output-root", default="result/metrics")
    parser.add_argument("--disable-tqdm", action="store_true")
    args = parser.parse_args()
    per_run, mean_std = write_structured_aggregate_tables(
        args.predictions_root,
        args.output_root,
        disable_tqdm=args.disable_tqdm,
    )
    print(f"Wrote {per_run}")
    print(f"Wrote {mean_std}")


if __name__ == "__main__":
    main()
