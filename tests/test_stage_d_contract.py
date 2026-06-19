from __future__ import annotations

from module.stage_a import InternalEvidenceExtractor
from module.stage_b import ExternalKnowledgeAcquisition
from module.stage_c import KnowledgeRelevanceFilterVerifier
from module.stage_d import EvidenceFusionReasoning


def test_stage_d_metadata_and_tensor_contract():
    stage_a, stage_c = _stage_inputs()
    output = EvidenceFusionReasoning()(stage_a, stage_c)

    assert output.shared_reasoning_state.shape == (256,)
    assert output.internal_memory.size(0) == len(stage_a.evidence_items)
    assert output.fused_tokens.shape == output.internal_memory.shape
    assert {"target", "intent", "tactic"} <= set(output.task_latents)
    assert output.metadata.internal_token_count == stage_a.internal_tokens.size(0)
    assert output.metadata.verified_knowledge_count == stage_c.verified_tokens.size(0)

    metadata = output.metadata
    assert metadata.regularizer_hook_mode == "detached_analysis_only"
    assert metadata.regularizer_hooks_are_differentiable is False
    assert metadata.analysis_hooks == metadata.regularizer_hooks
    assert isinstance(metadata.knowledge_origin_counts, dict)
    assert isinstance(metadata.knowledge_provenance_records, list)
    assert {"top_links", "has_attention"} <= set(metadata.attention_trace)
    assert "knowledge_need" in metadata.gate_statistics
    assert {"verified_knowledge_count", "task_support_used"} <= set(metadata.task_support_summary)
    assert metadata.support_matrix_columns == stage_c.metadata.support_matrix_columns
    assert "score_weights" in metadata.stage_c_policy

    assert len(metadata.knowledge_provenance_records) == len(stage_c.verified_items)
    for record in metadata.knowledge_provenance_records:
        assert {
            "knowledge_id",
            "candidate_origin",
            "is_external_knowledge",
            "is_generated",
            "is_fallback",
            "is_retrieved",
            "support_label",
            "support_score",
            "relevance_score",
            "validity_score",
            "final_score",
        } <= set(record)

    if metadata.attention_trace["has_attention"]:
        for link in metadata.attention_trace["top_links"]:
            assert {
                "evidence_id",
                "evidence_type",
                "knowledge_id",
                "candidate_origin",
                "attention_weight",
                "knowledge_final_score",
                "knowledge_support_label",
            } <= set(link)

    if stage_c.verified_items:
        summary = metadata.task_support_summary
        for task in ["target", "intent", "tactic"]:
            assert f"{task}_support_mean" in summary
            assert f"{task}_support_max" in summary
        assert any(
            key in metadata.gate_statistics
            for key in ["external_gate_mean", "token_level_mean", "sample_level"]
        )


def test_stage_d_prefers_knowledge_need_score_alias():
    stage_a, stage_c = _stage_inputs()
    stage_a.auxiliary_scores["knowledge_need_score"] = 0.9
    stage_a.auxiliary_scores["knowledge_need"] = 0.1

    output = EvidenceFusionReasoning()(stage_a, stage_c)

    assert abs(output.metadata.knowledge_need - 0.9) < 1e-6
    assert abs(output.metadata.gate_statistics["knowledge_need"] - 0.9) < 1e-6


def test_stage_d_zero_knowledge_diagnostic_contract():
    stage_a = InternalEvidenceExtractor()(
        {
            "sample_id": "stage_d_zero",
            "dataset_name": "facebook",
            "image_path": None,
            "ocr_text_full": "tiny",
        }
    )
    stage_b = ExternalKnowledgeAcquisition(corpus_paths=[], fallback_candidates=True)(stage_a)
    stage_c = KnowledgeRelevanceFilterVerifier(
        min_relevance=0.99,
        allow_low_relevance_fallback=False,
    )(stage_a, stage_b)
    output = EvidenceFusionReasoning()(stage_a, stage_c)

    assert output.cross_attention_weights.shape[1] == 0
    assert output.metadata.attention_trace["has_attention"] is False
    assert output.metadata.verified_knowledge_count == 0
    assert output.metadata.knowledge_provenance_records == []
    assert output.metadata.task_support_summary["task_support_used"] is False


def _stage_inputs():
    stage_a = InternalEvidenceExtractor()(
        {
            "sample_id": "stage_d_contract",
            "dataset_name": "facebook",
            "image_path": None,
            "ocr_text_full": "Trump meme joke about election context",
        }
    )
    stage_b = ExternalKnowledgeAcquisition(corpus_paths=[], fallback_candidates=True)(stage_a)
    stage_c = KnowledgeRelevanceFilterVerifier()(stage_a, stage_b)
    return stage_a, stage_c
