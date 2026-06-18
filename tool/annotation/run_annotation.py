"""CLI entry point for meme annotation."""

from __future__ import annotations

import argparse
import logging

from tool.annotation.config import DATASET_ROOT, MODEL_NAME, OUTPUT_DIR
from tool.annotation.llm_annotator import OpenAIAnnotatorClient
from tool.annotation.pipeline import AnnotationPipeline
from tool.annotation.utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate meme target, intent, and tactic with an LLM/VLM.")
    parser.add_argument("--dataset", required=True, choices=["harmc", "harmp", "facebook", "memotion"])
    parser.add_argument("--mode", required=True, choices=["multimodal", "text"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--debug_sample_id", default=None)
    parser.add_argument("--save_prompt", action="store_true")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger(level=logging.DEBUG if args.verbose else logging.INFO)

    annotator = None
    if not args.dry_run:
        annotator = OpenAIAnnotatorClient(model=args.model, multimodal=args.mode == "multimodal")

    pipeline = AnnotationPipeline(
        annotator=annotator,
        output_dir=OUTPUT_DIR,
        dataset_root=DATASET_ROOT,
    )
    pipeline.run(
        dataset_name=args.dataset,
        mode=args.mode,
        limit=args.limit,
        resume=args.resume,
        dry_run=args.dry_run,
        debug_sample_id=args.debug_sample_id,
        save_prompt=args.save_prompt,
    )


if __name__ == "__main__":
    main()
