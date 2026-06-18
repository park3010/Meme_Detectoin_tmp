"""Train/evaluate the image-only CLIP baseline."""

from __future__ import annotations

from run_baseline_text_only import run_cli


if __name__ == "__main__":
    run_cli("image_only_clip")
