from __future__ import annotations

from module.stage_a import InternalEvidenceExtractor
from module.stage_b import ExternalKnowledgeAcquisition
from module.stage_c import KnowledgeRelevanceFilterVerifier
from module.stage_d import EvidenceFusionReasoning
from module.stage_e import StructuredInterpretationHead
from module.stage_c.schemas import StageCMetadata, StageCOutput
import torch


def test_stage_e_forward_structured_output():
    stage_a = InternalEvidenceExtractor()({"sample_id": "s1", "dataset_name": "harm_p", "image_path": None, "ocr_text_full": "THIS IS A SHIT SHOW"})
    stage_b = ExternalKnowledgeAcquisition()(stage_a)
    stage_c = KnowledgeRelevanceFilterVerifier()(stage_a, stage_b)
    stage_d = EvidenceFusionReasoning()(stage_a, stage_c)
    output = StructuredInterpretationHead()(stage_a, stage_c, stage_d)
    assert output.structured_prediction["sample_id"] == "s1"
    assert "presence" in output.structured_prediction["target"]
    assert "background_knowledge_needed" in output.structured_prediction["intent"]
    assert "rhetorical" in output.structured_prediction["tactic"]
    assert output.harmfulness.logits is not None
    assert output.rationale


def test_stage_e_zero_external_evidence_schema_fields():
    stage_a = InternalEvidenceExtractor()({"sample_id": "s1", "dataset_name": "harm_p", "image_path": None, "ocr_text_full": "plain meme"})
    stage_c = StageCOutput(
        sample_id="s1",
        dataset_name="harm_p",
        verified_items=[],
        verified_tokens=torch.zeros(0, 256),
        support_matrix=torch.zeros(0, 6),
        final_scores=torch.zeros(0),
        internal_summary="plain meme",
        metadata=StageCMetadata(0, 0, 8, 0.05),
    )
    stage_d = EvidenceFusionReasoning()(stage_a, stage_c)
    output = StructuredInterpretationHead()(stage_a, stage_c, stage_d)
    assert output.supporting_evidence["external"] == []
    assert {"presence", "granularity", "attributes", "label_summary"} <= set(output.structured_prediction["target"])
    assert {"primary", "stance", "secondary", "background_knowledge_needed"} <= set(output.structured_prediction["intent"])
