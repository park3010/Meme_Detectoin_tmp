"""Run stage-wise ablation and fusion/gating comparison experiments."""

from __future__ import annotations

import argparse

from common import ROOT

from experiments.ablation_configs import ABLATION_MODES, FUSION_MODES
from experiments.ablation_runner import run_ablation_experiment, run_fusion_experiment
from experiments.splits import DEFAULT_SEEDS, normalize_dataset_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--dataset", nargs="+", default=["harm_c"])
    parser.add_argument("--ablation", nargs="+", default=["full"], help="Ablation name(s), or all.")
    parser.add_argument("--fusion-mode", nargs="+", default=None, help="Optional fusion mode(s), or all.")
    parser.add_argument("--seed", nargs="+", type=int, default=[42])
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--output-root", default="result")
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--print-components", action="store_true")
    args = parser.parse_args()

    datasets = normalize_dataset_names(args.dataset) or ["harm_c", "harm_p", "facebook", "memotion"]
    seeds = DEFAULT_SEEDS if args.all_seeds else args.seed
    ablations = ABLATION_MODES if "all" in args.ablation else args.ablation
    fusion_modes = []
    if args.fusion_mode:
        fusion_modes = FUSION_MODES if "all" in args.fusion_mode else args.fusion_mode

    for dataset in datasets:
        for seed in seeds:
            for ablation in ablations:
                metrics = run_ablation_experiment(
                    dataset,
                    ablation,
                    seed=seed,
                    config_path=args.config,
                    split_file=args.split_file,
                    output_root=args.output_root,
                    limit=args.limit,
                    disable_tqdm=args.disable_tqdm,
                    print_components=args.print_components,
                )
                print(f"{dataset}/ablation_{ablation}/seed={seed}: macro_f1={metrics.get('macro_f1')}")
            for mode in fusion_modes:
                metrics = run_fusion_experiment(
                    dataset,
                    mode,
                    seed=seed,
                    config_path=args.config,
                    split_file=args.split_file,
                    output_root=args.output_root,
                    limit=args.limit,
                    disable_tqdm=args.disable_tqdm,
                    print_components=args.print_components,
                )
                print(f"{dataset}/fusion_{mode}/seed={seed}: macro_f1={metrics.get('macro_f1')}")


if __name__ == "__main__":
    main()
