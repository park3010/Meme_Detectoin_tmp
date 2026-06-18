from __future__ import annotations

from module.stage_a import InternalEvidenceExtractor
from module.stage_b import ExternalKnowledgeAcquisition
from module.stage_b.external_knowledge_acquisition import collect_linkable_surface_forms


def test_stage_b_forward_from_stage_a():
    stage_a = InternalEvidenceExtractor()({"sample_id": "s1", "dataset_name": "harm_c", "image_path": None, "ocr_text_full": "Bernie memes"})
    stage_b = ExternalKnowledgeAcquisition()
    output = stage_b(stage_a)
    assert output.candidate_tokens.size(1) == 256
    assert output.knowledge_candidates


def test_stage_b_uses_stage_a_evidence_surfaces():
    stage_a = InternalEvidenceExtractor()(
        {
            "sample_id": "s1",
            "dataset_name": "harm_c",
            "image_path": None,
            "ocr_text_full": "China Stop eating everything that moves",
        }
    )
    surfaces, stats = collect_linkable_surface_forms(stage_a, "China Stop eating everything that moves")
    assert surfaces
    assert stats["visual_evidence_used"] >= 1

    output = ExternalKnowledgeAcquisition(corpus_paths=[], fallback_candidates=True)(stage_a)
    assert output.metadata.evidence_surface_count >= 1
    assert output.metadata.visual_evidence_used is True
    assert output.metadata.query_source_breakdown["meme_template"] >= 1
    assert output.candidate_tokens.shape[1] == 256
