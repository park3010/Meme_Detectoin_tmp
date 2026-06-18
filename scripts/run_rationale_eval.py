"""Run automatic rationale quality evaluation and export human-rating template."""

from __future__ import annotations

import argparse

from common import ROOT  # noqa: F401

from experiments.rationale_eval import run_rationale_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--model", default="ours_full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--disable-tqdm", action="store_true")
    args = parser.parse_args()
    rows = run_rationale_evaluation(
        args.dataset,
        args.model,
        args.seed,
        args.result_root,
        disable_tqdm=args.disable_tqdm,
    )
    print(f"Wrote rationale metrics for {len(rows)} samples.")


if __name__ == "__main__":
    main()
