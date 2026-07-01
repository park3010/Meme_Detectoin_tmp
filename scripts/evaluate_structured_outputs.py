"""Evaluate structured framework prediction JSONL files."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import ROOT  # noqa: F401

from experiments.evaluation import evaluate_prediction_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="ours_full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prediction-file", default=None)
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--output-root", default="result/metrics")
    parser.add_argument("--disable-tqdm", action="store_true")
    args = parser.parse_args()

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
    )
    print(f"{args.dataset}/{args.model}/seed={args.seed}: structured metrics saved to {args.output_root}")
    print(metrics)


if __name__ == "__main__":
    main()
