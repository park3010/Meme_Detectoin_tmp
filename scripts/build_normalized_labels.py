"""Build normalized silver-label JSONL files from annotation JSONL files."""

from __future__ import annotations

import argparse

from common import ROOT, print_json

from experiments.annotation_normalization import (
    build_normalized_dataset,
    load_normalization_config,
    resolve_dataset_names,
    summarize_normalization,
    write_normalized_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "annotation_normalization.yaml"))
    parser.add_argument("--dataset", default="all", help="all, harm_c, harm_p, facebook, or memotion")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--annotation-root", default=None)
    parser.add_argument("--normalized-root", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--clean-confidence", default="high,medium")
    parser.add_argument("--include-not-sure", action="store_true")
    parser.add_argument("--disable-tqdm", action="store_true")
    args = parser.parse_args()

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
    summary = summarize_normalization(rows, warnings) | {"outputs": {key: str(value) for key, value in outputs.items()}}
    print_json(summary)


if __name__ == "__main__":
    main()
