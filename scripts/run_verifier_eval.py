"""Evaluate Stage C verifier behavior with weak-label fallback."""

from __future__ import annotations

import argparse

from common import ROOT

from experiments.splits import normalize_dataset_names
from experiments.verifier_eval import run_verifier_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", default=["harm_c"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-root", default="result")
    parser.add_argument("--disable-tqdm", action="store_true")
    args = parser.parse_args()
    datasets = normalize_dataset_names(args.dataset) or ["harm_c", "harm_p", "facebook", "memotion"]
    for dataset in datasets:
        metrics = run_verifier_evaluation(
            dataset,
            seed=args.seed,
            config_path=args.config,
            output_root=args.output_root,
            limit=args.limit,
            disable_tqdm=args.disable_tqdm,
        )
        print(f"{dataset}: relevance_f1={metrics.get('relevance_f1')} support_macro_f1={metrics.get('support_macro_f1')}")


if __name__ == "__main__":
    main()
