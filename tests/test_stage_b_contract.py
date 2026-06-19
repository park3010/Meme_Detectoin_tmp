from __future__ import annotations

from module.stage_a import InternalEvidenceExtractor
from module.stage_b import ExternalKnowledgeAcquisition
from module.stage_b.external_knowledge_acquisition import (
    collect_linkable_surface_forms,
    collect_linkable_surface_records,
)


def test_stage_b_candidate_and_metadata_contract():
    stage_a = _stage_a_output()
    output = ExternalKnowledgeAcquisition(corpus_paths=[], fallback_candidates=True)(stage_a)

    assert len(output.knowledge_candidates) == output.candidate_tokens.size(0)
    for index, candidate in enumerate(output.knowledge_candidates):
        assert candidate.token_index == index
        assert 0 <= candidate.token_index < output.candidate_tokens.size(0)
        metadata = candidate.metadata
        assert metadata["source_stage"] == "stage_b"
        assert metadata["requires_verification"] is True
        assert {
            "candidate_origin",
            "is_retrieved",
            "is_fallback",
            "is_generated",
            "is_external_knowledge",
        } <= set(metadata)
        if candidate.source == "fallback":
            assert metadata["candidate_origin"] == "fallback"
            assert metadata["is_external_knowledge"] is False
        if candidate.candidate_type == "generated_hypothesis":
            assert metadata["candidate_origin"] == "generated_hypothesis"
            assert metadata["is_generated"] is True
            assert metadata["is_external_knowledge"] is False
            assert metadata["is_interpretive_hypothesis"] is True
        if candidate.candidate_type == "retrieved" and candidate.source != "fallback":
            assert metadata["candidate_origin"] == "retrieved"
            assert metadata["is_retrieved"] is True

    metadata = output.metadata
    assert metadata.surface_records
    assert metadata.query_records
    assert metadata.query_source_breakdown
    assert metadata.evidence_surface_count >= 1
    assert metadata.retrieval_stats
    assert len(metadata.query_records) == metadata.query_count


def test_stage_b_surface_record_contract_and_alias_preference():
    stage_a = _stage_a_output()
    stage_a.metadata.auxiliary_labels["stage_a_multimodal_relation"] = "cross_modal_implication"
    stage_a.metadata.auxiliary_labels["multimodal_relation"] = "old_value_should_not_win"

    surfaces, stats = collect_linkable_surface_forms(stage_a, "China Stop eating everything that moves")
    records, record_stats = collect_linkable_surface_records(stage_a, "China Stop eating everything that moves")

    assert surfaces
    assert records
    assert stats["visual_evidence_used"] >= 1
    assert record_stats["visual_evidence_used"] >= 1
    assert any(record["source_stage"] == "stage_a" for record in records)
    assert any(record["surface"] == "multimodal relation cross_modal_implication" for record in records)
    for record in records:
        assert record["surface"]
        assert record["surface_type"]
        assert record["source_stage"]
        assert "is_heuristic" in record

    output = ExternalKnowledgeAcquisition(corpus_paths=[], fallback_candidates=True)(stage_a)
    assert any("cross_modal_implication" in record["query"] for record in output.metadata.query_records)


def _stage_a_output():
    return InternalEvidenceExtractor()(
        {
            "sample_id": "stage_b_contract",
            "dataset_name": "harm_c",
            "image_path": None,
            "ocr_text_full": "China Stop eating everything that moves",
        }
    )
