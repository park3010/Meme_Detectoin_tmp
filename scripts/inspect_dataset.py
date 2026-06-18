"""Inspect unified dataset loading and annotation joins."""

from __future__ import annotations

import argparse

from common import ROOT, print_json

from dataset import MemeDataset
from utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--dataset", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    dataset = MemeDataset(
        dataset_root=cfg.get("paths", {}).get("dataset_root", "dataset/V1"),
        annotation_root=cfg.get("paths", {}).get("annotation_root", "outputs"),
        dataset_names=args.dataset,
        keep_missing_images=True,
        limit=args.limit,
    )
    print_json({"statistics": dataset.statistics(), "preview": dataset.preview(args.limit)})


if __name__ == "__main__":
    main()
