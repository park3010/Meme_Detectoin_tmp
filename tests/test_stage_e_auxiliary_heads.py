from __future__ import annotations

from module.losses import StructuredMemeLoss, extract_supervision_from_annotation
from module.stage_a import InternalEvidenceExtractor
from module.stage_b import ExternalKnowledgeAcquisition
from module.stage_c import KnowledgeRelevanceFilterVerifier
from module.stage_d import EvidenceFusionReasoning
from module.stage_e import StructuredInterpretationHead


def test_stage_e_auxiliary_heads_emit_trainable_payloads():
    output = _stage_e_output()
    structured = output.structured_prediction
    target = structured["target"]
    tactic = structured["tactic"]

    assert target["presence"] in ["explicit", "implicit", "none"]
    assert target["presence_scores"]
    assert target["presence_logits"].requires_grad
    assert target["presence_source"] == "target_presence_head"
    assert target["presence_provenance"] == "logits_aux"
    assert {"heuristic_presence", "heuristic_presence_score"} <= set(target)

    assert tactic["multimodal_relation"] in StructuredInterpretationHead().tactic.relation_labels
    assert tactic["multimodal_relation_scores"]
    assert tactic["multimodal_relation_logits"].requires_grad
    assert tactic["multimodal_relation_source"] == "tactic_multimodal_relation_head"
    assert tactic["multimodal_relation_provenance"] == "logits_aux"
    assert tactic["stage_a_multimodal_relation"] == "cross_modal_implication"

    hooks = structured["training_hooks"]
    assert hooks["target_presence_logits"].requires_grad
    assert hooks["target_presence_scores"]
    assert hooks["tactic_multimodal_relation_logits"].requires_grad
    assert hooks["tactic_multimodal_relation_scores"]
    assert structured["output_provenance"]["field_provenance"]["target.presence"] == "logits_aux"
    assert structured["output_provenance"]["field_provenance"]["tactic.multimodal_relation"] == "logits_aux"


def test_auxiliary_losses_are_differentiable_and_backward():
    output = _stage_e_output()
    losses = StructuredMemeLoss()(
        output,
        {
            "target_presence": "explicit",
            "tactic_multimodal_relation": "cross_modal_implication",
        },
    )

    assert losses["target_presence"].requires_grad is True
    assert losses["tactic_multimodal_relation"].requires_grad is True
    losses["total"].backward()
    assert output.structured_prediction["target"]["presence_logits"].grad_fn is not None
    assert output.structured_prediction["tactic"]["multimodal_relation_logits"].grad_fn is not None


def test_auxiliary_loss_fallback_and_ignored_labels():
    output = _stage_e_output()
    target = output.structured_prediction["target"]
    tactic = output.structured_prediction["tactic"]
    target["presence_logits"] = None
    tactic["multimodal_relation_logits"] = None

    fallback = StructuredMemeLoss()(
        output,
        {
            "target_presence": "explicit",
            "tactic_multimodal_relation": "cross_modal_implication",
        },
    )
    assert fallback["target_presence"].requires_grad is False
    assert fallback["tactic_multimodal_relation"].requires_grad is False

    ignored = StructuredMemeLoss()(
        output,
        {
            "target_presence": "ambiguous",
            "tactic_multimodal_relation": "unknown",
        },
    )
    assert "target_presence" not in ignored
    assert "tactic_multimodal_relation" not in ignored


def test_structured_style_auxiliary_gold_aliases_are_extracted():
    supervision = extract_supervision_from_annotation(
        {
            "target": {"presence": "explicit", "granularity": "community"},
            "tactic": {"multimodal_relation": "cross_modal_implication"},
        }
    )

    assert supervision["target_presence"] == "explicit"
    assert supervision["target_granularity"] == "community"
    assert supervision["tactic_multimodal_relation"] == "cross_modal_implication"


def _stage_e_output():
    stage_a = InternalEvidenceExtractor()(
        {
            "sample_id": "stage_e_aux",
            "dataset_name": "harm_c",
            "image_path": None,
            "ocr_text_full": "China implies a different context, yeah right",
        }
    )
    stage_a.metadata.auxiliary_labels["stage_a_multimodal_relation"] = "cross_modal_implication"
    stage_b = ExternalKnowledgeAcquisition(corpus_paths=[], fallback_candidates=True)(stage_a)
    stage_c = KnowledgeRelevanceFilterVerifier()(stage_a, stage_b)
    stage_d = EvidenceFusionReasoning()(stage_a, stage_c)
    return StructuredInterpretationHead()(stage_a, stage_c, stage_d)
