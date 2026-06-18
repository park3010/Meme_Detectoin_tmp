"""Run the full 5-stage meme interpretation pipeline."""

from __future__ import annotations

import argparse

from common import add_common_args
from module.pipeline.runner import PipelineRunner


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--run-until", choices=["a", "b", "c", "d", "e"], default="e")
    args = parser.parse_args()
    runner = PipelineRunner(args.config)
    runner.run(dataset_names=args.dataset, limit=args.limit, run_until=args.run_until, save=not args.no_save)


if __name__ == "__main__":
    main()
