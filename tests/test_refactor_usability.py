from __future__ import annotations

from module.baselines.classifier_heads import MLPClassifierHead
from module.baselines.clip_text_concat import CLIPTextConcatClassifier
from module.baselines.image_only_clip import ImageOnlyCLIPClassifier
from module.baselines.models import TextOnlyEncoderClassifier
from module.stage_a.extractor import InternalEvidenceExtractor
from module.stage_b.acquisition import ExternalKnowledgeAcquisition
from module.stage_c.verifier import KnowledgeRelevanceFilterVerifier
from module.stage_d.fusion import EvidenceFusionReasoning
from module.stage_e.interpretation import StructuredInterpretationHead
from experiments.components import print_pipeline_components
from module.pipeline.model import HarmfulMemePipeline


def test_old_and_new_import_paths_work():
    assert ImageOnlyCLIPClassifier.model_name == "image_only_clip"
    assert TextOnlyEncoderClassifier.model_name == "text_only_encoder"
    assert CLIPTextConcatClassifier.model_name == "clip_text_concat"
    assert MLPClassifierHead(2).net is not None
    assert InternalEvidenceExtractor is not None
    assert ExternalKnowledgeAcquisition is not None
    assert KnowledgeRelevanceFilterVerifier is not None
    assert EvidenceFusionReasoning is not None
    assert StructuredInterpretationHead is not None


def test_print_pipeline_components(capsys):
    pipeline = HarmfulMemePipeline().eval()
    print_pipeline_components(pipeline)
    captured = capsys.readouterr()
    assert "Pipeline components:" in captured.out
    assert "module.stage_a.extractor.InternalEvidenceExtractor" in captured.out
