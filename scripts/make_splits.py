"""Create or load deterministic train/valid/test splits."""

from __future__ import annotations

import argparse

from common import ROOT

from dataset import MemeDataset
from experiments.splits import DEFAULT_SEEDS, build_splits_for_dataset, normalize_dataset_names, save_splits
from utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--dataset", nargs="+", default=["all"])
    parser.add_argument("--seed", nargs="+", type=int, default=[42])
    parser.add_argument("--all-seeds", action="store_true", help="Generate the default five paper seeds.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--output-root", default="result/splits")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    dataset_names = normalize_dataset_names(args.dataset)
    dataset = MemeDataset(
        dataset_root=cfg.get("paths", {}).get("dataset_root", "dataset/source"),
        annotation_root=cfg.get("paths", {}).get("annotation_root", "dataset/annotation"),
        dataset_names=dataset_names,
        keep_missing_images=True,
        limit=args.limit,
    )
    names = sorted({sample.dataset_name for sample in dataset.samples})
    seeds = DEFAULT_SEEDS if args.all_seeds else args.seed
    for name in names:
        for seed in seeds:
            splits = build_splits_for_dataset(
                name,
                dataset,
                seed=seed,
                train_ratio=args.train_ratio,
                valid_ratio=args.valid_ratio,
                test_ratio=args.test_ratio,
                dataset_root=cfg.get("paths", {}).get("dataset_root", "dataset/source"),
            )
            path = save_splits(splits, name, seed, args.output_root)
            print(f"Saved {name} seed={seed}: {path} train={len(splits['train'])} valid={len(splits['valid'])} test={len(splits['test'])}")


if __name__ == "__main__":
    main()
