"""Pipeline wiring for the modular meme interpretation scaffold."""

from module.pipeline.model import HarmfulMemePipeline
from module.pipeline.runner import PipelineRunner

__all__ = ["HarmfulMemePipeline", "PipelineRunner"]
