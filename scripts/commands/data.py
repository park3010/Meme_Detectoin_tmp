"""Data-preparation subcommands for the unified CLI."""

from __future__ import annotations

import argparse
from typing import Any

from dataset import MemeDataset
from experiments.dataset_stats import compute_dataset_statistics, format_statistics_summary, save_dataset_statistics
from experiments.splits import DEFAULT_SEEDS, build_splits_for_dataset, normalize_dataset_names, save_splits
from utils.io import load_yaml


def add_data_commands(subparsers: argparse._SubParsersAction, default_config: str) -> None:
    data = subparsers.add_parser("data", help="Dataset statistics, splits, and label inspection utilities.")
    data_subparsers = data.add_subparsers(dest="data_command", required=True)

    stats = data_subparsers.add_parser("dataset-stats", help="Generate dataset statistics tables.")
    stats.add_argument("--config", default=default_config)
    stats.add_argument("--dataset", nargs="+", default=["all"])
    stats.add_argument("--output-root", default="result/dataset_stats")
    stats.set_defaults(func=_cmd_dataset_stats)

    splits = data_subparsers.add_parser("make-splits", help="Create deterministic train/valid/test splits.")
    splits.add_argument("--config", default=default_config)
    splits.add_argument("--dataset", nargs="+", default=["all"])
    splits.add_argument("--seed", nargs="+", type=int, default=[42])
    splits.add_argument("--all-seeds", action="store_true")
    splits.add_argument("--train-ratio", type=float, default=0.7)
    splits.add_argument("--valid-ratio", type=float, default=0.1)
    splits.add_argument("--test-ratio", type=float, default=0.2)
    splits.add_argument("--output-root", default="result/splits")
    splits.add_argument("--limit", type=int, default=None)
    splits.set_defaults(func=_cmd_make_splits)


def _cmd_dataset_stats(args: argparse.Namespace) -> None:
    cfg = load_yaml(args.config)
    datasets = None if "all" in args.dataset else args.dataset
    rows = compute_dataset_statistics(
        dataset_root=cfg.get("paths", {}).get("dataset_root", "dataset/source"),
        annotation_root=cfg.get("paths", {}).get("annotation_root", "dataset/annotation"),
        dataset_names=datasets,
    )
    json_path, csv_path = save_dataset_statistics(rows, args.output_root)
    print(format_statistics_summary(rows))
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")


def _cmd_make_splits(args: argparse.Namespace) -> None:
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


__all__ = ["add_data_commands"]
