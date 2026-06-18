from __future__ import annotations

import torch

from module.stage_e.intent_head import IntentHead
from module.stage_e.tactic_head import TacticHead
from utils.io import load_yaml


def test_intent_head_covers_non_ignored_vocab_labels():
    vocab = load_yaml("configs/label_vocab.yaml")
    spec = vocab["single_label_fields"]["intent_primary"]
    expected = set(spec["labels"]) - set(spec.get("ignore_labels", []))

    assert expected <= set(IntentHead.labels)
    assert "unknown" not in IntentHead.labels


def test_tactic_head_covers_non_ignored_vocab_labels():
    vocab = load_yaml("configs/label_vocab.yaml")
    spec = vocab["multi_label_fields"]["tactic_rhetorical"]
    expected = set(spec["labels"]) - set(spec.get("ignore_labels", []))

    assert expected <= set(TacticHead.labels)
    assert "unknown" not in TacticHead.labels


def test_intent_head_logit_size_matches_label_space():
    head = IntentHead(hidden_dim=256)
    logits = head.compute_logits(torch.zeros(256), "join the march and criticize corruption")

    assert logits.shape[-1] == len(IntentHead.labels)


def test_tactic_head_logit_size_matches_label_space():
    head = TacticHead(hidden_dim=256)
    logits = head.compute_logits(torch.zeros(256), "what about the hoax vs reality?")

    assert logits.shape[-1] == len(TacticHead.labels)
