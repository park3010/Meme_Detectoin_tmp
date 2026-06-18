"""Inspect normalized-label coverage, targets, and vocab metadata."""

from __future__ import annotations

import argparse

from common import print_json

from dataset import LabelVocab, NormalizedMemeDataset
from dataset.label_adapter import _source_dataset_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="all", help="all, harm_c, harm_p, facebook, or memotion")
    parser.add_argument("--label-set", choices=["full", "clean"], default="full")
    parser.add_argument("--dataset-root", default="dataset/source")
    parser.add_argument("--annotation-root", default="dataset/annotation")
    parser.add_argument("--normalized-root", default="dataset/annotation_normalized")
    parser.add_argument("--vocab", default="configs/label_vocab.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--require-normalized-label", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-vocab", action="store_true")
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
