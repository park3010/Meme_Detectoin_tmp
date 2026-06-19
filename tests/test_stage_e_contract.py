from __future__ import annotations

from module.stage_a import InternalEvidenceExtractor
from module.stage_b import ExternalKnowledgeAcquisition
from module.stage_c import KnowledgeRelevanceFilterVerifier
from module.stage_d import EvidenceFusionReasoning
from module.stage_e import StructuredInterpretationHead


def test_stage_e_structured_output_provenance_contract():
    stage_a = InternalEvidenceExtractor()(
        {
            "sample_id": "stage_e_contract",
            "dataset_name": "harm_c",
            "image_path": None,
            "ocr_text_full": "China meme says YEAH RIGHT about election context",
        }
    )
    stage_a.auxiliary_scores["knowledge_need_score"] = 0.9
    stage_a.auxiliary_scores["knowledge_need"] = 0.1
    stage_a.metadata.auxiliary_labels["stage_a_multimodal_relation"] = "cross_modal_implication"
    stage_a.metadata.auxiliary_labels["multimodal_relation"] = "old_value_should_not_win"

    stage_b = ExternalKnowledgeAcquisition(corpus_paths=[], fallback_candidates=True)(stage_a)
    stage_c = KnowledgeRelevanceFilterVerifier()(stage_a, stage_b)
    stage_d = EvidenceFusionReasoning()(stage_a, stage_c)
    output = StructuredInterpretationHead()(stage_a, stage_c, stage_d)

    metadata = output.metadata
    assert metadata.output_contract_version == "stage_e_structured_output_v1"
    assert metadata.field_provenance["harmfulness.label"] == "logits"
    assert metadata.field_provenance["target.granularity"] == "logits"
    assert metadata.field_provenance["target.presence"] == "heuristic_proxy"
    assert metadata.field_provenance["intent.primary"] == "logits"
    assert metadata.field_provenance["tactic.multimodal_relation"] == "stage_a_cue_proxy"
    assert metadata.field_provenance["rationale"] == "template"
    assert metadata.trainable_logits_fields
    assert metadata.proxy_fields
    assert metadata.template_fields == ["rationale"]
    assert metadata.cue_fields
    assert metadata.label_spaces

    structured = output.structured_prediction
    provenance = structured["output_provenance"]
    assert provenance["field_provenance"] == metadata.field_provenance
    hooks = structured["training_hooks"]
    assert "field_provenance" in hooks
    assert "trainable_logits_fields" in hooks
    assert "proxy_fields" in hooks
    assert "target_presence_logits" not in hooks
    assert "tactic_multimodal_relation_logits" not in hooks

    intent = structured["intent"]
    assert abs(intent["background_knowledge_score"] - 0.9) < 1e-6
    assert intent["background_knowledge_provenance"] == "cue_proxy"

    tactic = structured["tactic"]
    assert tactic["multimodal_relation"] == "cross_modal_implication"
    assert tactic["stage_a_multimodal_relation"] == "cross_modal_implication"
    assert tactic["multimodal_relation_provenance"] == "stage_a_cue_proxy"
    assert tactic["rhetorical_primary"] == output.tactic.label
    assert tactic["rhetorical_provenance"] == "logits_multilabel_or_top1_rendered"

    target = structured["target"]
    assert target["presence_provenance"] == "heuristic_proxy"
    assert {
        "presence_source",
        "heuristic_presence",
        "heuristic_presence_score",
    } <= set(target)

    for item in output.supporting_evidence["internal"]:
        assert item["attribution_backend"] == "gate_attention_score_proxy"
        assert item["source_stage"] == "stage_a"
        assert {"modality", "grounding_type", "is_heuristic"} <= set(item)
    for item in output.supporting_evidence["external"]:
        assert item["attribution_backend"] == "final_score_attention_support_proxy"
        assert {
            "candidate_origin",
            "is_external_knowledge",
            "verification_status",
        } <= set(item)
