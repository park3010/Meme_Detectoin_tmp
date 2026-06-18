"""Shared utilities for the annotation pipeline."""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Iterable

from tool.annotation.config import IMAGE_EXTENSIONS


def setup_logger(name: str = "meme_annotator", level: int = logging.INFO) -> logging.Logger:
    """Create a console logger with a compact formatter."""

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(handler)
    return logger


def safe_normalize_string(value: Any) -> str:
    """Normalize arbitrary values into a clean single-line-ish string."""

    if value is None:
        return ""
    text = str(value)
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_image_file(path: Path) -> bool:
    """Return True if path has a supported image extension."""

    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON object to a JSONL file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield valid JSON objects from a JSONL file, skipping malformed lines."""

    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def load_processed_ids(path: Path) -> set[str]:
    """Load sample IDs already present in an annotation JSONL file."""

    processed: set[str] = set()
    for record in read_jsonl(path) or []:
        sample_id = record.get("sample_id")
        if isinstance(sample_id, str):
            processed.add(sample_id)
    return processed


def pydantic_to_dict(model: Any) -> dict[str, Any]:
    """Convert pydantic v1 or v2 models to dictionaries."""

    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()
