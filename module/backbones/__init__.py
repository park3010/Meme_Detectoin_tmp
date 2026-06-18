"""Backbone wrappers and adapter interfaces."""

from module.backbones.clip_wrapper import CLIPWrapper
from module.backbones.cross_encoder_adapter import CrossEncoderAdapter
from module.backbones.detector_adapter import Detection, DetectorAdapter
from module.backbones.generator_adapter import GeneratorAdapter
from module.backbones.retriever_adapter import KnowledgeDocument, LocalRetrieverAdapter
from module.backbones.text_encoder_wrapper import TextEncoderWrapper

__all__ = [
    "CLIPWrapper",
    "CrossEncoderAdapter",
    "Detection",
    "DetectorAdapter",
    "GeneratorAdapter",
    "KnowledgeDocument",
    "LocalRetrieverAdapter",
    "TextEncoderWrapper",
]
