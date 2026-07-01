from __future__ import annotations

from experiments import structured_eval
from experiments.train import _normalize_sample
from module.losses import extract_supervision_from_annotation


def test_extract_supervision_prefers_normalized_targets_over_legacy():
    sample = _normalized_sample()
    sample["annotation"] = {
        "target": {"target_presence": "explicit", "protected_attribute": ["legacy"]},
        "intent": {"intent_primary": "legacy_intent"},
    }

    supervision = extract_supervision_from_annotation(sample)

    assert supervision["harmfulness"] == "harmful"
    assert supervision["target_presence"] == "implicit"
    assert supervision["target_attributes"] == ["religion", "nationality"]
    assert supervision["intent_primary"] == "ridicule_mockery"
    assert supervision["sample_weight"] == 0.5


def test_zero_masks_exclude_fields_from_supervision():
    sample = _normalized_sample()
    sample["targets"]["masks"]["target_presence"] = 0
    sample["targets"]["masks"]["protected_attribute"] = 0

    supervision = extract_supervision_from_annotation(sample)

    assert "target_presence" not in supervision
    assert "target_attributes" not in supervision
    assert supervision["harmfulness"] == "harmful"


def test_normalized_evidence_text_is_collected_as_list():
    supervision = extract_supervision_from_annotation(_normalized_sample())

    assert "mocking caption" in supervision["evidence_text"]
    assert "visual cue" in supervision["evidence_text"]
    assert all(isinstance(item, str) for item in supervision["evidence_text"])


def test_structured_eval_none_and_bool_helpers():
    assert structured_eval._is_missing_label("none") is False
    assert structured_eval._is_missing_label("unknown") is True
    assert structured_eval._bool_label("False") == "false"
    assert structured_eval._bool_label("yes") == "true"


def test_baseline_normalize_sample_reads_normalized_harmfulness():
    sample = _normalized_sample()
    sample["raw_label"] = 0

    normalized = _normalize_sample(sample)

    assert normalized["label"] == 1


def test_baseline_normalize_sample_falls_back_to_raw_label_when_masked():
    sample = _normalized_sample()
    sample["raw_label"] = 0
    sample["targets"]["masks"]["harmfulness"] = 0

    normalized = _normalize_sample(sample)

    assert normalized["label"] == 0


def _normalized_sample() -> dict:
    label_strings = {
        "harmfulness": "harmful",
        "target_presence": "implicit",
        "target_granularity": "community",
        "protected_attribute": ["religion", "nationality"],
        "intent_primary": "ridicule_mockery",
        "secondary_intent": ["criticism"],
        "stance": "hostile",
        "background_knowledge_needed": False,
        "tactic_rhetorical": ["stereotype", "sarcasm_irony"],
        "tactic_multimodal_relation": "cross_modal_implication",
    }
    masks = {
        "harmfulness": 1,
        "target_presence": 1,
        "target_granularity": 1,
        "protected_attribute": 1,
        "intent_primary": 1,
        "secondary_intent": 1,
        "stance": 1,
        "background_knowledge_needed": 1,
        "tactic_rhetorical": 1,
        "tactic_multimodal_relation": 1,
    }
    evidence_text = {
        "target_text_span": "mocking caption",
        "target_visual_cue": "visual cue",
        "key_text_evidence": "",
    }
    return {
        "sample_id": "s1",
        "dataset_name": "facebook",
        "raw_label": 0,
        "targets": {
            "label_strings": label_strings,
            "class_ids": {"harmfulness": 1},
            "masks": masks,
            "sample_weight": 0.5,
            "evidence_text": evidence_text,
        },
        "label_strings": label_strings,
        "evidence_text": evidence_text,
        "sample_weight": 0.5,
        "annotation": {},
    }
