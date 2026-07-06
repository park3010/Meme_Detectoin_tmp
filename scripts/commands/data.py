"""Data-preparation subcommands for the unified CLI."""

from __future__ import annotations

import argparse

from common import ROOT, print_json
from dataset import LabelVocab, MemeDataset, NormalizedMemeDataset
from dataset.labels import _source_dataset_names
from experiments.data_preparation import (
    build_normalized_dataset,
    compute_dataset_statistics,
    format_audit_summary,
    format_statistics_summary,
    load_normalization_config,
    resolve_dataset_names,
    run_annotation_audit,
    save_dataset_statistics,
    summarize_normalization,
    write_normalized_outputs,
)
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

    normalize = data_subparsers.add_parser("normalize-labels", help="Build normalized silver-label JSONL files.")
    normalize.add_argument("--config", default=str(ROOT / "configs" / "annotation_normalization.yaml"))
    normalize.add_argument("--dataset", default="all")
    normalize.add_argument("--dataset-root", default=None)
    normalize.add_argument("--annotation-root", default=None)
    normalize.add_argument("--normalized-root", default=None)
    normalize.add_argument("--limit", type=int, default=None)
    normalize.add_argument("--clean-confidence", default="high,medium")
    normalize.add_argument("--include-not-sure", action="store_true")
    normalize.add_argument("--disable-tqdm", action="store_true")
    normalize.set_defaults(func=_cmd_normalize_labels)

    audit = data_subparsers.add_parser("audit-annotations", help="Audit annotation and normalized-label coverage.")
    audit.add_argument("--config", default=str(ROOT / "configs" / "annotation_normalization.yaml"))
    audit.add_argument("--dataset", default="all")
    audit.add_argument("--dataset-root", default=None)
    audit.add_argument("--annotation-root", default=None)
    audit.add_argument("--audit-root", default=None)
    audit.add_argument("--limit", type=int, default=None)
    audit.add_argument("--max-examples-per-flag", type=int, default=50)
    audit.add_argument("--disable-tqdm", action="store_true")
    audit.set_defaults(func=_cmd_audit_annotations)

    inspect_dataset = data_subparsers.add_parser("inspect-dataset", help="Inspect unified dataset loading.")
    inspect_dataset.add_argument("--config", default=default_config)
    inspect_dataset.add_argument("--dataset", nargs="*", default=None)
    inspect_dataset.add_argument("--limit", type=int, default=5)
    inspect_dataset.set_defaults(func=_cmd_inspect_dataset)

    inspect_labels = data_subparsers.add_parser("inspect-labels", help="Inspect normalized-label coverage and vocab metadata.")
    inspect_labels.add_argument("--dataset", default="all")
    inspect_labels.add_argument("--label-set", choices=["full", "clean"], default="full")
    inspect_labels.add_argument("--dataset-root", default="dataset/source")
    inspect_labels.add_argument("--annotation-root", default="dataset/annotation")
    inspect_labels.add_argument("--normalized-root", default="dataset/annotation_normalized")
    inspect_labels.add_argument("--vocab", default="configs/label_vocab.yaml")
    inspect_labels.add_argument("--limit", type=int, default=None)
    inspect_labels.add_argument("--require-normalized-label", action=argparse.BooleanOptionalAction, default=True)
    inspect_labels.add_argument("--print-vocab", action="store_true")
    inspect_labels.add_argument("--config", default=default_config)
    inspect_labels.set_defaults(func=_cmd_inspect_labels)


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


def _cmd_normalize_labels(args: argparse.Namespace) -> None:
    cfg = load_normalization_config(args.config)
    paths = cfg.get("paths", {})
    dataset_root = args.dataset_root or paths.get("dataset_root", "dataset/source")
    annotation_root = args.annotation_root or paths.get("annotation_root", "dataset/annotation")
    normalized_root = args.normalized_root or paths.get("normalized_root", "dataset/annotation_normalized")
    dataset_names = resolve_dataset_names(args.dataset, cfg)
    clean_confidence = {item.strip() for item in args.clean_confidence.split(",") if item.strip()}
    rows, warnings = build_normalized_dataset(
        dataset_names=dataset_names,
        dataset_root=dataset_root,
        annotation_root=annotation_root,
        config=cfg,
        limit=args.limit,
        disable_tqdm=args.disable_tqdm,
    )
    outputs = write_normalized_outputs(
        rows,
        warnings,
        normalized_root,
        clean_confidence=clean_confidence,
        include_not_sure=args.include_not_sure,
    )
    print_json(summarize_normalization(rows, warnings) | {"outputs": {key: str(value) for key, value in outputs.items()}})


def _cmd_audit_annotations(args: argparse.Namespace) -> None:
    cfg = load_normalization_config(args.config)
    paths = cfg.get("paths", {})
    dataset_names = resolve_dataset_names(args.dataset, cfg)
    result = run_annotation_audit(
        dataset_names=dataset_names,
        dataset_root=args.dataset_root or paths.get("dataset_root", "dataset/source"),
        annotation_root=args.annotation_root or paths.get("annotation_root", "dataset/annotation"),
        config=cfg,
        audit_root=args.audit_root or paths.get("audit_root", "result/annotation_audit"),
        limit=args.limit,
        max_examples_per_flag=args.max_examples_per_flag,
        disable_tqdm=args.disable_tqdm,
    )
    print(format_audit_summary(result))


def _cmd_inspect_dataset(args: argparse.Namespace) -> None:
    cfg = load_yaml(args.config)
    dataset = MemeDataset(
        dataset_root=cfg.get("paths", {}).get("dataset_root", "dataset/source"),
        annotation_root=cfg.get("paths", {}).get("annotation_root", "dataset/annotation"),
        dataset_names=args.dataset,
        keep_missing_images=True,
        limit=args.limit,
    )
    print_json({"statistics": dataset.statistics(), "preview": dataset.preview(args.limit)})


def _cmd_inspect_labels(args: argparse.Namespace) -> None:
    requested = None if args.dataset == "all" else [args.dataset]
    dataset_names = _source_dataset_names(["all"] if args.dataset == "all" else requested)
    dataset = NormalizedMemeDataset(
        dataset_root=args.dataset_root,
        annotation_root=args.annotation_root,
        normalized_root=args.normalized_root,
        dataset_names=["all"] if args.dataset == "all" else requested,
        label_set=args.label_set,
        vocab_path=args.vocab,
        keep_missing_images=True,
        limit=args.limit,
        require_normalized_label=args.require_normalized_label,
    )
    vocab = LabelVocab.from_yaml(args.vocab)
    stats = dataset.statistics()
    first_targets = dataset[0].get("targets", {}) if len(dataset) else {}
    summary = {
        "loaded_samples": len(dataset),
        "label_set": args.label_set,
        "datasets": dataset_names,
        "coverage": stats.get("coverage", {}),
        "preview": dataset.preview(min(args.limit or 5, 5)),
        "target_key_names": {
            key: sorted(value.keys()) if isinstance(value, dict) else []
            for key, value in first_targets.items()
            if key in {"class_ids", "multi_hot", "binary", "masks"}
        },
        "vocab_sizes": {
            **{field: vocab.num_classes(field) for field in vocab.single_label_fields},
            **{field: vocab.num_classes(field) for field in vocab.multi_label_fields},
            **{field: 2 for field in vocab.binary_fields},
        },
        "sample_weight": stats.get("sample_weight", {}),
    }
    if args.print_vocab:
        summary["vocab"] = {
            "single_label_fields": vocab.single_label_fields,
            "multi_label_fields": vocab.multi_label_fields,
            "binary_fields": vocab.binary_fields,
            "optional_text_fields": vocab.optional_text_fields,
            "sample_weight": vocab.sample_weight_config,
        }
    print_json(summary)


__all__ = ["add_data_commands"]
