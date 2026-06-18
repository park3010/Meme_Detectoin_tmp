"""Configuration defaults for the meme annotation pipeline."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = Path(os.getenv("MEME_DATASET_ROOT", str(PROJECT_ROOT / "dataset" / "source"))).expanduser()
OUTPUT_DIR = Path(os.getenv("MEME_ANNOTATION_OUTPUT_DIR", str(PROJECT_ROOT / "dataset" / "annotation"))).expanduser()

DATASET_FOLDERS: dict[str, str] = {
    "harmc": "covid_img+text",
    "harmp": "political_img+text",
    "facebook": "facebook_img+text",
    "memotion": "memotion_img+text",
}

DATASET_DISPLAY_NAMES: dict[str, str] = {
    "harmc": "Harm-C",
    "harmp": "Harm-P",
    "facebook": "Facebook Hateful Memes",
    "memotion": "Memotion",
}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("MEME_ANNOTATION_MODEL", "gpt-5.5")

MAX_RETRIES = int(os.getenv("MEME_ANNOTATION_MAX_RETRIES", "3"))
TIMEOUT_SECONDS = float(os.getenv("MEME_ANNOTATION_TIMEOUT_SECONDS", "60"))
RETRY_BACKOFF_SECONDS = float(os.getenv("MEME_ANNOTATION_RETRY_BACKOFF_SECONDS", "2"))
MAX_OUTPUT_TOKENS = int(os.getenv("MEME_ANNOTATION_MAX_OUTPUT_TOKENS", "2048"))

BATCH_SIZE = int(os.getenv("MEME_ANNOTATION_BATCH_SIZE", "1"))
CHECKPOINT_EVERY = int(os.getenv("MEME_ANNOTATION_CHECKPOINT_EVERY", "10"))

MULTIMODAL_MODE = os.getenv("MEME_ANNOTATION_MULTIMODAL", "1").lower() in {"1", "true", "yes"}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
