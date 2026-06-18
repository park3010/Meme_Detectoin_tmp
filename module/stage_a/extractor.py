"""Canonical Stage A implementation imports.

The detailed Stage A components remain in focused files to keep behavior stable.
This module gives readers one obvious import target for the complete extractor.
"""

from module.stage_a.incongruity_analyzer import CrossModalIncongruityAnalyzer
from module.stage_a.internal_evidence_extractor import InternalEvidenceExtractor
from module.stage_a.local_symbol_extractor import LocalObjectSymbolExtractor
from module.stage_a.text_semantic_encoder import TextSemanticEncoder
from module.stage_a.visual_encoder import GlobalVisualEncoder

__all__ = [
    "GlobalVisualEncoder",
    "TextSemanticEncoder",
    "LocalObjectSymbolExtractor",
    "CrossModalIncongruityAnalyzer",
    "InternalEvidenceExtractor",
]
