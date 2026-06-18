from __future__ import annotations

from module.stage_a import InternalEvidenceExtractor
from module.stage_b import ExternalKnowledgeAcquisition
from module.stage_c import KnowledgeRelevanceFilterVerifier


def test_stage_c_forward_from_candidates():
    stage_a = InternalEvidenceExtractor()({"sample_id": "s1", "dataset_name": "harm_c", "image_path": None, "ocr_text_full": "Trump meme joke"})
    stage_b = ExternalKnowledgeAcquisition()(stage_a)
    output = KnowledgeRelevanceFilterVerifier()(stage_a, stage_b)
    assert output.verified_tokens.size(1) == 256
    assert output.metadata.input_candidate_count >= output.metadata.filtered_candidate_count
    assert output.support_matrix.shape[1] == 6
    assert output.task_support_matrix.shape[1] == 3
    assert output.metadata.support_matrix_columns == [
        "relevance",
        "target_support",
        "intent_support",
        "tactic_support",
        "validity",
        "final",
    ]


def test_stage_c_low_relevance_fallback_filtering():
    stage_a = InternalEvidenceExtractor()({"sample_id": "s1", "dataset_name": "harm_c", "image_path": None, "ocr_text_full": "x"})
    stage_b = ExternalKnowledgeAcquisition(corpus_paths=[], fallback_candidates=True)(stage_a)
    output = KnowledgeRelevanceFilterVerifier(min_relevance=0.99, allow_low_relevance_fallback=False)(stage_a, stage_b)
    assert output.metadata.allow_low_relevance_fallback is False
    assert len(output.verified_items) == 0
