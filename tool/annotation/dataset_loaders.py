"""Dataset loaders for prepared all.jsonl meme OCR datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from tool.annotation.config import DATASET_FOLDERS, IMAGE_EXTENSIONS
from tool.annotation.schemas import UnifiedSample
from tool.annotation.utils import safe_normalize_string, setup_logger


logger = setup_logger(__name__)


def _parse_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one JSON object per line, skipping malformed lines safely."""

    records: list[dict[str, Any]] = []
    if not path.exists():
        logger.warning("OCR file does not exist: %s", path)
        return records

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSONL line %s in %s: %s", line_no, path, exc)
                continue
            if not isinstance(obj, dict):
                logger.warning("Skipping non-object JSONL line %s in %s", line_no, path)
                continue
            records.append(obj)
    return records


def _resolve_image_path(dataset_dir: Path, image_value: Any, sample_id: str) -> Optional[Path]:
    """Resolve an image path from the image field, falling back to sample_id stem."""

    img_dir = dataset_dir / "img"
    if isinstance(image_value, str) and image_value.strip():
        candidate = img_dir / Path(image_value).name
        if candidate.exists():
            return candidate
        logger.warning("Image field did not resolve for sample_id=%s: %s", sample_id, candidate)

    for ext in sorted(IMAGE_EXTENSIONS):
        fallback = img_dir / f"{sample_id}{ext}"
        if fallback.exists():
            return fallback
    return None


def load_all_jsonl_samples(
    dataset_name: str,
    dataset_dir: Path,
    keep_missing_images: bool = False,
) -> list[UnifiedSample]:
    """Load samples from the prepared `{dataset_dir}/txt/all.jsonl` format.

    Only the `text` field is treated as OCR input. The `labels` field is
    preserved for storage/auditing as raw_label but is never used by prompts.
    """

    all_jsonl_path = dataset_dir / "txt" / "all.jsonl"
    records = _parse_jsonl(all_jsonl_path)
    samples: list[UnifiedSample] = []

    for index, record in enumerate(records, start=1):
        raw_id = record.get("id")
        if raw_id is None or safe_normalize_string(raw_id) == "":
            logger.warning("Skipping record without id at line-like index %s in %s", index, all_jsonl_path)
            continue

        sample_id = safe_normalize_string(raw_id)
        ocr_text = safe_normalize_string(record.get("text", ""))
        image_path = _resolve_image_path(dataset_dir, record.get("image"), sample_id)

        if image_path is None:
            logger.warning("Missing image for sample_id=%s in dataset=%s", sample_id, dataset_name)
            if not keep_missing_images:
                continue

        samples.append(
            UnifiedSample(
                sample_id=sample_id,
                dataset_name=dataset_name,
                image_path=str(image_path) if image_path else None,
                ocr_text=ocr_text,
                raw_label=record.get("labels"),
                raw_record=record,
            )
        )

    logger.info("Loaded %s samples from %s", len(samples), all_jsonl_path)
    return samples


def load_harmc_samples(root_dir: Path, keep_missing_images: bool = False) -> list[UnifiedSample]:
    """Load Harm-C samples from covid_img+text/txt/all.jsonl."""

    return load_all_jsonl_samples("harmc", root_dir / DATASET_FOLDERS["harmc"], keep_missing_images)


def load_harmp_samples(root_dir: Path, keep_missing_images: bool = False) -> list[UnifiedSample]:
    """Load Harm-P samples from political_img+text/txt/all.jsonl."""

    return load_all_jsonl_samples("harmp", root_dir / DATASET_FOLDERS["harmp"], keep_missing_images)


def load_facebook_samples(root_dir: Path, keep_missing_images: bool = False) -> list[UnifiedSample]:
    """Load Facebook Hateful Memes samples from facebook_img+text/txt/all.jsonl."""

    return load_all_jsonl_samples("facebook", root_dir / DATASET_FOLDERS["facebook"], keep_missing_images)


def load_memotion_samples(root_dir: Path, keep_missing_images: bool = False) -> list[UnifiedSample]:
    """Load Memotion samples from memotion_img+text/txt/all.jsonl."""

    return load_all_jsonl_samples("memotion", root_dir / DATASET_FOLDERS["memotion"], keep_missing_images)


def load_samples(dataset_name: str, root_dir: Path, keep_missing_images: bool = False) -> list[UnifiedSample]:
    """Load one supported prepared dataset by short name."""

    loaders = {
        "harmc": load_harmc_samples,
        "harmp": load_harmp_samples,
        "facebook": load_facebook_samples,
        "memotion": load_memotion_samples,
    }
    if dataset_name not in loaders:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return loaders[dataset_name](root_dir, keep_missing_images)
