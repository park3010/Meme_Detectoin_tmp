from __future__ import annotations

from module.stage_a import InternalEvidenceExtractor
from module.stage_b import ExternalKnowledgeAcquisition
from module.stage_c import KnowledgeRelevanceFilterVerifier
from module.stage_c.knowledge_filter_verifier import _build_claims


def test_stage_c_metadata_alignment_and_verified_provenance():
    stage_a, stage_b = _stage_inputs()
    output = KnowledgeRelevanceFilterVerifier()(stage_a, stage_b)

    metadata = output.metadata
    assert metadata.score_weights == {
        "relevance": 0.46,
        "support": 0.24,
        "validity": 0.20,
        "retrieval_prior": 0.10,
    }
    assert metadata.label_bonus["support"] == 0.10
    assert isinstance(metadata.input_origin_counts, dict)
    assert isinstance(metadata.verified_origin_counts, dict)
    assert isinstance(metadata.rejection_records, list)
    assert metadata.rejected_count == len(metadata.rejection_records)
    assert metadata.verification_policy["policy_type"] == "lightweight_candidate_filter"

    assert len(output.verified_items) == output.verified_tokens.size(0)
    assert len(output.verified_items) == output.support_matrix.size(0)
    assert len(output.verified_items) == output.final_scores.size(0)
    assert output.support_matrix.shape[1] == 6
    for index, item in enumerate(output.verified_items):
        assert item.token_index == index
        assert item.metadata["verification_status"] == "accepted"
        assert {
            "candidate_origin",
            "is_external_knowledge",
            "is_generated",
            "is_fallback",
            "is_retrieved",
            "requires_verification",
            "score_policy",
        } <= set(item.metadata)
        assert item.metadata["score_policy"]["weights"]["relevance"] == 0.46
        assert "raw_score_components" in item.metadata


def test_stage_c_low_relevance_rejection_trace():
    stage_a, stage_b = _stage_inputs()
    output = KnowledgeRelevanceFilterVerifier(
        min_relevance=0.99,
        allow_low_relevance_fallback=False,
    )(stage_a, stage_b)

    assert output.metadata.rejected_count == len(output.metadata.rejection_records)
    assert output.metadata.rejected_count >= 0
    if output.metadata.rejection_records:
        record = output.metadata.rejection_records[0]
        assert record["reason"] == "low_relevance"
        assert {
            "candidate_id",
            "candidate_origin",
            "relevance_score",
            "min_relevance",
            "is_external_knowledge",
        } <= set(record)


def test_stage_c_claims_prefer_stage_a_relation_alias():
    stage_a, _ = _stage_inputs()
    stage_a.metadata.auxiliary_labels["stage_a_multimodal_relation"] = "cross_modal_implication"
    stage_a.metadata.auxiliary_labels["multimodal_relation"] = "old_value_should_not_win"

    claims = _build_claims(stage_a)

    assert "cross_modal_implication" in claims["tactic"]
    assert "old_value_should_not_win" not in claims["tactic"]


def _stage_inputs():
    stage_a = InternalEvidenceExtractor()(
        {
            "sample_id": "stage_c_contract",
            "dataset_name": "harm_c",
            "image_path": None,
            "ocr_text_full": "Trump meme joke about an election",
        }
    )
    stage_b = ExternalKnowledgeAcquisition(corpus_paths=[], fallback_candidates=True)(stage_a)
    return stage_a, stage_b
