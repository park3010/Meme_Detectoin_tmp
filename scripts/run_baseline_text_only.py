"""Train/evaluate a simple baseline classifier."""

from __future__ import annotations

import argparse

from common import ROOT

from experiments.splits import DEFAULT_SEEDS, normalize_dataset_names
from experiments.train_baseline import BaselineRunConfig, run_baseline_experiment


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser shared by all simple baseline runners."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--dataset", nargs="+", default=["harm_c"])
    parser.add_argument("--seed", nargs="+", type=int, default=[42])
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--early-stop-metric", default="val_macro_f1")
    parser.add_argument("--early-stop-mode", choices=["max", "min"], default="max")
    parser.add_argument("--save-best", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-last", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--output-root", default="result")
    parser.add_argument("--label-set", choices=["full", "clean"], default="full")
    parser.add_argument("--normalized-root", default="dataset/annotation_normalized")
    parser.add_argument("--vocab-path", default="configs/label_vocab.yaml")
    parser.add_argument("--use-normalized-labels", action="store_true", default=False)
    parser.add_argument("--require-normalized-label", dest="require_normalized_label", action="store_true", default=True)
    parser.add_argument("--allow-missing-normalized-label", dest="require_normalized_label", action="store_false")
    parser.add_argument("--use-sample-weight", action="store_true", default=False)
    return parser


def config_from_args(args: argparse.Namespace, model_name: str, dataset: str, seed: int) -> BaselineRunConfig:
    """Create a BaselineRunConfig from parsed CLI arguments."""

    return BaselineRunConfig(
        model_name=model_name,
        dataset_name=dataset,
        seed=seed,
        config_path=args.config,
        split_file=args.split_file,
        output_root=args.output_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        min_delta=args.min_delta,
        early_stop_metric=args.early_stop_metric,
        early_stop_mode=args.early_stop_mode,
        save_best=args.save_best,
        save_last=args.save_last,
        disable_tqdm=args.disable_tqdm,
        device=args.device,
        limit=args.limit,
        normalized_root=args.normalized_root,
        label_set=args.label_set,
        vocab_path=args.vocab_path,
        use_normalized_labels=args.use_normalized_labels,
        require_normalized_label=args.require_normalized_label,
        use_sample_weight=args.use_sample_weight,
    )


def run_cli(model_name: str = "text_only_encoder") -> None:
    parser = build_parser()
    args = parser.parse_args()

    datasets = normalize_dataset_names(args.dataset) or ["harm_c", "harm_p", "facebook", "memotion"]
    seeds = DEFAULT_SEEDS if args.all_seeds else args.seed
    for dataset in datasets:
        for seed in seeds:
            cfg = config_from_args(args, model_name, dataset, seed)
            metrics = run_baseline_experiment(cfg)
            print(f"{dataset}/{model_name}/seed={seed}: macro_f1={metrics.get('macro_f1')} accuracy={metrics.get('accuracy')}")


if __name__ == "__main__":
    run_cli("text_only_encoder")
