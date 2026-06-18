from __future__ import annotations

import torch

from module.losses import StructuredMemeLoss, is_differentiable_loss, loss_provenance


def test_loss_provenance_mapping():
    assert loss_provenance("harmfulness") == "logits"
    assert loss_provenance("target_granularity") == "logits"
    assert loss_provenance("intent_primary") == "logits"
    assert loss_provenance("tactic_rhetorical") == "logits_multilabel"
    assert loss_provenance("target_presence") == "proxy_detached_score"
    assert loss_provenance("stance") == "proxy_rule_score"
    assert loss_provenance("consistency") == "proxy_detached_metadata"


def test_differentiability_expectation_uses_provenance_and_tensor_state():
    logits_loss = torch.tensor(1.0, requires_grad=True)
    detached_proxy = torch.tensor(0.5)

    assert is_differentiable_loss("harmfulness", logits_loss) is True
    assert is_differentiable_loss("target_presence", detached_proxy) is False
    assert is_differentiable_loss("harmfulness", logits_loss.detach()) is False


def test_structured_loss_descriptions_are_logging_only():
    losses = {
        "harmfulness": torch.tensor(1.0, requires_grad=True),
        "target_presence": torch.tensor(0.5),
        "total": torch.tensor(1.5, requires_grad=True),
    }

    descriptions = StructuredMemeLoss().describe_losses(losses)

    assert descriptions["harmfulness"] == {
        "value": 1.0,
        "provenance": "logits",
        "differentiable_expected": True,
        "requires_grad": True,
    }
    assert descriptions["target_presence"]["provenance"] == "proxy_detached_score"
    assert descriptions["target_presence"]["differentiable_expected"] is False
    assert descriptions["target_presence"]["requires_grad"] is False
    assert losses["harmfulness"].requires_grad is True
