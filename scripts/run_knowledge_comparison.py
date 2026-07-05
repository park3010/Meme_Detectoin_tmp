"""Run knowledge-source comparison modes for the full framework."""

from __future__ import annotations

import argparse

from common import ROOT

from experiments.ablation_configs import KNOWLEDGE_MODES
from experiments.knowledge_comparison import run_knowledge_comparison
from experiments.splits import DEFAULT_SEEDS, normalize_dataset_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--dataset", nargs="+", default=["harm_c"])
    parser.add_argument("--mode", nargs="+", default=["verified"], help="Knowledge mode(s), or all.")
    parser.add_argument("--seed", nargs="+", type=int, default=[42])
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--output-root", default="result")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--print-components", action="store_true")
    args = parser.parse_args()

    datasets = normalize_dataset_names(args.dataset) or ["harm_c", "harm_p", "facebook", "memotion"]
    modes = KNOWLEDGE_MODES if "all" in args.mode else args.mode
    seeds = DEFAULT_SEEDS if args.all_seeds else args.seed
    for dataset in datasets:
        for seed in seeds:
            for mode in modes:
                metrics = run_knowledge_comparison(
                    dataset,
                    mode,
                    seed=seed,
                    config_path=args.config,
                    split_file=args.split_file,
                    output_root=args.output_root,
                    limit=args.limit,
                    disable_tqdm=args.disable_tqdm,
                    print_components=args.print_components,
                    device=args.device,
                )
                print(f"{dataset}/knowledge_{mode}/seed={seed}: macro_f1={metrics.get('macro_f1')}")


if __name__ == "__main__":
    main()
