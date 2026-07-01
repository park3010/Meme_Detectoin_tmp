from __future__ import annotations

import torch

from module.losses import (
    StructuredMemeLoss,
    classification_loss_from_logits,
    extract_supervision_from_annotation,
    multilabel_loss_from_logits,
)
from module.runner import HarmfulMemePipeline
from utils.eval_utils import binary_classification_metrics, evidence_precision_recall_at_k, multiclass_metrics


def test_phase2_losses_and_metrics_smoke():
    pipeline = HarmfulMemePipeline().eval()
    outputs = pipeline(
        {
            "sample_id": "s1",
            "dataset_name": "harm_c",
            "image_path": None,
            "ocr_text_full": "THIS IS A SHIT SHOW ABOUT AMERICA",
        }
    )
    losses = StructuredMemeLoss()(outputs["stage_e"], {"harmfulness": "harmful"})
    assert "total" in losses
    assert losses["total"].item() >= 0
    assert losses["harmfulness"].requires_grad

    binary = binary_classification_metrics([1, 0, 1], [1, 0, 0])
    assert binary["accuracy"] >= 0
    multi = multiclass_metrics(["a", "b"], ["a", "a"])
    assert "confusion" in multi
    evidence = evidence_precision_recall_at_k(["e1", "e2"], {"e2"}, k=2)
    assert evidence["recall_at_k"] == 1.0


def test_nested_annotation_supervision_and_logits_losses():
    annotation = {
        "raw_label": 1,
        "annotation": {
            "target": {"target_presence": "explicit", "target_granularity": "community", "protected_attribute": ["nationality"]},
            "intent": {"intent_primary": "ridicule_mockery", "stance": "hostile", "secondary_intent": "self_expression", "background_knowledge_needed": True},
            "tactic": {"tactic_rhetorical": ["sarcasm_irony"], "tactic_multimodal_relation": "incongruent"},
            "evidence": {"key_text_evidence": "mocking text", "key_visual_evidence": "visual cue"},
        },
    }
    supervision = extract_supervision_from_annotation(annotation)
    assert supervision["harmfulness"] == "harmful"
    assert supervision["target_granularity"] == "community"
    assert supervision["tactic_rhetorical"] == ["sarcasm_irony"]
    assert supervision["evidence_text"] == ["mocking text", "visual cue"]

    logits = torch.randn(3, requires_grad=True)
    loss = classification_loss_from_logits(logits, 1)
    assert loss.requires_grad
    multi = multilabel_loss_from_logits(logits, [0, 2])
    assert multi.requires_grad
