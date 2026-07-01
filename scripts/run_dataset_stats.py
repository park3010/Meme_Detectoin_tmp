"""Generate dataset statistics for paper Table 1."""

from __future__ import annotations

import argparse

from common import ROOT

from experiments.dataset_stats import compute_dataset_statistics, format_statistics_summary, save_dataset_statistics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--dataset", nargs="+", default=["all"], help="all or one/more of harm_c harm_p facebook memotion")
    parser.add_argument("--output-root", default="result/dataset_stats")
    args = parser.parse_args()

    from utils.io import load_yaml

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


if __name__ == "__main__":
    main()
