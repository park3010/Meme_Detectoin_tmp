"""Run cross-dataset robustness experiments."""

from __future__ import annotations

import argparse

from common import ROOT

from experiments.cross_domain import CrossDomainConfig, run_cross_domain


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", required=True, choices=["in_domain", "train_one_test_others", "mixed_train", "leave_one_domain_out"])
    parser.add_argument("--model", default="ours_full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--train-dataset", default=None)
    parser.add_argument("--heldout", default=None)
    parser.add_argument("--test-dataset", default=None)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-root", default="result")
    parser.add_argument("--disable-tqdm", action="store_true")
    args = parser.parse_args()
    rows = run_cross_domain(
        CrossDomainConfig(
            setting=args.setting,
            model_name=args.model,
            seed=args.seed,
            config_path=args.config,
            train_dataset=args.train_dataset,
            heldout=args.heldout,
            test_dataset=args.test_dataset,
            epochs=args.epochs,
            lr=args.lr,
            device=args.device,
            limit=args.limit,
            output_root=args.output_root,
            disable_tqdm=args.disable_tqdm,
        )
    )
    print(f"Wrote {len(rows)} cross-domain row(s).")


if __name__ == "__main__":
    main()
