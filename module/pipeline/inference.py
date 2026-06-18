"""Convenience inference helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from module.pipeline.runner import PipelineRunner


def run_single_sample(sample: dict[str, Any], config_path: str | Path = "configs/default.yaml", run_until: str = "e") -> dict[str, Any]:
    """Run one in-memory sample through the pipeline."""

    runner = PipelineRunner(config_path=config_path)
    return runner.pipeline(sample, run_until=run_until)
