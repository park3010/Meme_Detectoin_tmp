import json

import pytest
import torch

from experiments.source_sanity import (
    assert_source_only_dataset_names,
    deterministic_manifest_hash,
    deterministic_shuffle,
    parameter_update,
    reject_forbidden_manifest,
)


def test_source_allowlist_accepts_harmeme_domains():
    assert_source_only_dataset_names(["harm_c", "harm_p"])


@pytest.mark.parametrize("name", ["facebook", "fhm", "memotion", "all"])
def test_source_allowlist_rejects_non_source_names(name):
    with pytest.raises(ValueError, match="permits only"):
        assert_source_only_dataset_names([name])


@pytest.mark.parametrize("row", [
    {"dataset_name": "facebook", "sample_id": "1"},
    {"dataset_family": "fhm", "sample_id": "1"},
    {"dataset_name": "memotion", "sample_id": "1"},
])
def test_forbidden_manifest_rejection(row):
    with pytest.raises(ValueError, match="forbidden"):
        reject_forbidden_manifest([row])


def test_manifest_hash_is_deterministic():
    rows = [{"sample_key": "harm_c::a"}, {"sample_key": "harm_p::b"}]
    assert deterministic_manifest_hash(rows) == deterministic_manifest_hash(rows)
    assert deterministic_manifest_hash(rows) != deterministic_manifest_hash(list(reversed(rows)))


def test_shuffle_is_deterministic_and_does_not_mutate_input():
    values = [0, 1, 0, 1, 1, 0]
    original = list(values)
    first, digest = deterministic_shuffle(values)
    second, second_digest = deterministic_shuffle(values)
    assert values == original
    assert first == second and digest == second_digest
    assert sorted(first) == sorted(values)


def test_parameter_update_detection_and_finite_status():
    before = torch.tensor([1.0, 2.0])
    result = parameter_update(before, before + 0.1)
    assert result["updated"] and result["finite"]
    assert result["update_norm"] > 0


def test_zero_parameter_update_is_detected():
    value = torch.tensor([1.0])
    assert not parameter_update(value, value.clone())["updated"]


def test_nan_parameter_update_is_not_finite():
    result = parameter_update(torch.tensor([1.0]), torch.tensor([float("nan")]))
    assert not result["finite"]


def test_tokenizer_policy_is_not_changed_by_source_module():
    source = open("experiments/source_sanity.py", encoding="utf-8").read()
    assert "fix_mistral_regex" not in source
