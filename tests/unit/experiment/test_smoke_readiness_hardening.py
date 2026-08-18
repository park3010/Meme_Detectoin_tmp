import json
from pathlib import Path

import pytest
import torch

from experiments.evaluation import compute_harmfulness_metrics
from experiments.formal_tasks import FORMAL_TASKS
from experiments.smoke_diagnostics import auroc_polarity, canonical_prediction_hash, collapse_checks, index_predictions
from experiments.storage_safety import atomic_torch_save, resolve_mount, storage_preflight


def test_alignment_uses_ids_and_hash_is_order_independent():
    rows = [{"sample_id": "b", "x": 2}, {"sample_id": "a", "x": 1}]
    assert list(index_predictions(rows)) == ["b", "a"]
    assert canonical_prediction_hash(rows) == canonical_prediction_hash(list(reversed(rows)))


def test_duplicate_ids_fail():
    with pytest.raises(ValueError, match="duplicate"):
        index_predictions([{"sample_id": "a"}, {"sample_id": "a"}])


def test_harmful_score_polarity_and_inverted_auc():
    rows = [
        {"gold_label": 0, "prob_harmful": .1, "harmfulness_score": .9},
        {"gold_label": 1, "prob_harmful": .9, "harmfulness_score": .9},
    ]
    result = auroc_polarity(rows, "x", "validation")
    assert result["auroc_declared_harmful_score"] == 1
    assert result["auroc_one_minus_declared"] == 0
    assert result["legacy_field_is_predicted_class_confidence"]


def test_single_class_gold_auc_is_not_applicable():
    assert compute_harmfulness_metrics([1, 1], [1, 1], [.8, .9])["roc_auc"] is None


def test_collapse_nan_is_blocker_and_single_class_is_warning():
    row = {"gold_harmfulness":"harmful","pred_harmfulness":"harmful","gold_target":{},"target":{},"gold_intent":{},"intent":{},"gold_tactic":{},"tactic":{},"training_hooks":{"harmfulness_logits":[float("nan")]},"tactic_rhetorical_label_order":[]}
    checks, _, _ = collapse_checks([row], "x", "validation")
    assert any(x["status"] == "blocker" and x["reason"] == "nan_or_inf_logits" for x in checks)
    assert any(x["status"] == "warning" and x["reason"] == "single_class_prediction" for x in checks)


def test_formal_allowlist_excludes_protected_attribute():
    assert "protected_attribute" not in FORMAL_TASKS


def test_low_space_preflight_blocks(tmp_path):
    with pytest.raises(RuntimeError, match="storage preflight"):
        storage_preflight(tmp_path, estimated_checkpoint_bytes=10**30)


def test_atomic_save_replaces_and_loads(tmp_path):
    path = tmp_path / "best.pt"
    atomic_torch_save({"value": 1}, path)
    atomic_torch_save({"value": 2}, path)
    assert torch.load(path, weights_only=True)["value"] == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_failed_save_cleans_temp_and_preserves_final(tmp_path):
    path = tmp_path / "best.pt"
    path.write_bytes(b"old")
    def fail(_obj, handle):
        handle.write(b"partial")
        raise OSError("disk full")
    with pytest.raises(OSError):
        atomic_torch_save({}, path, save_fn=fail)
    assert path.read_bytes() == b"old"
    assert not list(tmp_path.glob("*.tmp"))


def test_mount_resolution(tmp_path):
    mount, filesystem = resolve_mount(tmp_path)
    assert mount.startswith("/") and filesystem
