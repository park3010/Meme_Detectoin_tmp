from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from module.baseline import MLPClassifierHead
from module.baseline import CLIPTextConcatClassifier
from module.baseline import ImageOnlyCLIPClassifier
from module.baseline import TextOnlyEncoderClassifier
from module.internal_evidence_extractor import InternalEvidenceExtractor
from module.external_knowledge_acquisition import ExternalKnowledgeAcquisition
from module.knowledge_filter_verifier import KnowledgeRelevanceFilterVerifier
from module.evidence_fusion_reasoning import EvidenceFusionReasoning
from module.structured_interpretation_head import StructuredInterpretationHead
from experiments.components import print_pipeline_components
from module.runner import HarmfulMemePipeline
import run


def test_consolidated_public_imports_work():
    assert ImageOnlyCLIPClassifier.model_name == "image_only_clip"
    assert TextOnlyEncoderClassifier.model_name == "text_only_encoder"
    assert CLIPTextConcatClassifier.model_name == "clip_text_concat"
    assert MLPClassifierHead(2).net is not None
    assert InternalEvidenceExtractor is not None
    assert ExternalKnowledgeAcquisition is not None
    assert KnowledgeRelevanceFilterVerifier is not None
    assert EvidenceFusionReasoning is not None
    assert StructuredInterpretationHead is not None


def test_unified_cli_exposes_required_subcommands():
    parser = run.build_parser()
    for command in ["train", "baseline", "stage", "evaluate", "ablation", "audit"]:
        assert command in parser.format_help()


def test_required_consolidated_files_exist():
    for relative in [
        "module/internal_evidence_extractor.py",
        "module/external_knowledge_acquisition.py",
        "module/knowledge_filter_verifier.py",
        "module/evidence_fusion_reasoning.py",
        "module/structured_interpretation_head.py",
        "module/baseline.py",
        "module/losses.py",
        "module/runner.py",
        "module/backbone/vision.py",
        "module/backbone/text.py",
        "module/backbone/retrieval.py",
        "module/backbone/generation.py",
        "configs/config.yaml",
    ]:
        assert (ROOT / relative).exists()


def test_print_pipeline_components(capsys):
    pipeline = HarmfulMemePipeline().eval()
    print_pipeline_components(pipeline)
    captured = capsys.readouterr()
    assert "Pipeline components:" in captured.out
    assert "module.internal_evidence_extractor.InternalEvidenceExtractor" in captured.out
