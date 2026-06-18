"""Batch annotation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from tqdm import tqdm

from tool.annotation.config import CHECKPOINT_EVERY, DATASET_ROOT, OUTPUT_DIR
from tool.annotation.dataset_loaders import load_samples
from tool.annotation.llm_annotator import BaseAnnotatorClient
from tool.annotation.prompt_builder import SYSTEM_PROMPT, build_user_prompt
from tool.annotation.schemas import UnifiedSample
from tool.annotation.utils import load_processed_ids, pydantic_to_dict, setup_logger, write_jsonl


logger = setup_logger(__name__)


class AnnotationPipeline:
    """Run annotation over a dataset and write incremental JSONL outputs."""

    def __init__(
        self,
        annotator: Optional[BaseAnnotatorClient],
        output_dir: Path = OUTPUT_DIR,
        dataset_root: Path = DATASET_ROOT,
    ) -> None:
        self.annotator = annotator
        self.output_dir = output_dir
        self.dataset_root = dataset_root

    def run(
        self,
        dataset_name: str,
        mode: str,
        limit: Optional[int] = None,
        resume: bool = False,
        dry_run: bool = False,
        debug_sample_id: Optional[str] = None,
        save_prompt: bool = False,
    ) -> None:
        samples = load_samples(dataset_name, self.dataset_root, keep_missing_images=(mode == "text"))
        if debug_sample_id:
            samples = [sample for sample in samples if sample.sample_id == debug_sample_id]
        if limit is not None:
            samples = samples[:limit]

        logger.info("Loaded %s samples for dataset=%s", len(samples), dataset_name)

        if dry_run:
            self._dry_run(samples)
            return

        if self.annotator is None:
            raise ValueError("annotator is required unless dry_run=True")

        paths = self._output_paths(dataset_name)
        processed_ids = load_processed_ids(paths["annotations"]) if resume else set()
        multimodal = mode == "multimodal"

        completed = 0
        for sample in tqdm(samples, desc=f"Annotating {dataset_name}"):
            if resume and sample.sample_id in processed_ids:
                continue

            try:
                user_prompt = build_user_prompt(sample, multimodal=multimodal)
                result = self.annotator.annotate(sample, SYSTEM_PROMPT, user_prompt)
                raw_record = {"sample_id": sample.sample_id, "raw_response": result.raw_response}
                if save_prompt:
                    raw_record["prompt"] = user_prompt
                write_jsonl(paths["raw"], raw_record)

                if result.annotation is not None:
                    write_jsonl(
                        paths["annotations"],
                        {
                            "sample_id": sample.sample_id,
                            "dataset_name": sample.dataset_name,
                            "image_path": sample.image_path,
                            "raw_label": sample.raw_label,
                            "annotation": pydantic_to_dict(result.annotation),
                        },
                    )
                else:
                    write_jsonl(
                        paths["errors"],
                        {
                            "sample_id": sample.sample_id,
                            "error_type": result.error_type or "AnnotationError",
                            "error_message": result.error or "Unknown annotation failure",
                            "raw_response": result.raw_response,
                        },
                    )
            except Exception as exc:
                logger.exception("Unhandled per-sample failure for %s", sample.sample_id)
                write_jsonl(
                    paths["errors"],
                    {
                        "sample_id": sample.sample_id,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "raw_response": "",
                    },
                )

            completed += 1
            if completed % CHECKPOINT_EVERY == 0:
                logger.info("Checkpoint: processed %s new samples", completed)

    def _output_paths(self, dataset_name: str) -> dict[str, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "annotations": self.output_dir / f"{dataset_name}_annotations.jsonl",
            "raw": self.output_dir / f"{dataset_name}_raw_responses.jsonl",
            "errors": self.output_dir / f"{dataset_name}_errors.jsonl",
        }

    @staticmethod
    def _dry_run(samples: list[UnifiedSample], preview_count: int = 5) -> None:
        logger.info("Dry run only: API will not be called.")
        logger.info("Matched samples: %s", len(samples))
        for sample in samples[:preview_count]:
            logger.info(
                "sample_id=%s dataset=%s image=%s raw_label=%r text=%r",
                sample.sample_id,
                sample.dataset_name,
                sample.image_path,
                sample.raw_label,
                sample.ocr_text[:300],
            )
