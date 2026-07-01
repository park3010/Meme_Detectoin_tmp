"""Measure stage latency, resource use, and fallback rates."""

from __future__ import annotations

import argparse

from common import ROOT

from experiments.runtime_cost import run_runtime_cost_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="harm_c")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--output-root", default="result")
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--print-components", action="store_true")
    args = parser.parse_args()
    summary = run_runtime_cost_analysis(
        args.dataset,
        args.limit,
        args.device,
        args.warmup,
        args.config,
        args.output_root,
        disable_tqdm=args.disable_tqdm,
        print_components=args.print_components,
    )
    print(summary)


if __name__ == "__main__":
    main()
