"""Backbone adapter package."""

from module.backbone.generation import GeneratorAdapter
from module.backbone.retrieval import CrossEncoderAdapter, KnowledgeDocument, LocalRetrieverAdapter
from module.backbone.text import TextEncoderWrapper
from module.backbone.vision import CLIPWrapper, Detection, DetectorAdapter

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
