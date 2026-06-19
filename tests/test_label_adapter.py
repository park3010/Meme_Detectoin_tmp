from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from dataset import LabelVocab, NormalizedLabelAdapter, NormalizedLabelStore, NormalizedMemeDataset
from utils.io import read_jsonl


def test_normalized_label_store_indexes_by_dataset_and_sample(tmp_path):
    source, annotation_root, normalized_root = _write_mini_normalized_dataset(tmp_path)
    _ = source, annotation_root
    store = NormalizedLabelStore(normalized_root, dataset_names=["facebook"], label_set="full")

    assert len(store) == 2
    assert store.get("facebook", "a") is not None
    assert store.get("harm_c", "a") is None
    assert store.coverage_for_samples([{"dataset_name": "facebook", "sample_id": "a"}])["coverage_ratio"] == 1.0


def test_label_vocab_single_and_multihot_mapping():
    vocab = LabelVocab.from_yaml("configs/label_vocab.yaml")

    assert vocab.label_to_id("harmfulness", "harmful") == 1
    assert vocab.id_to_label("intent_primary", 0) == "ridicule_mockery"
    vector = vocab.multi_hot("tactic_rhetorical", ["stereotype", "sarcasm_irony", "unknown"])
    assert vector[vocab.label_to_id("tactic_rhetorical", "stereotype")] == 1
    assert vector[vocab.label_to_id("tactic_rhetorical", "sarcasm_irony")] == 1
    assert vector[vocab.label_to_id("tactic_rhetorical", "unknown")] == 0


def test_unknown_labels_have_ignore_masks_and_weights(tmp_path):
    _, _, normalized_root = _write_mini_normalized_dataset(tmp_path)
    store = NormalizedLabelStore(normalized_root, dataset_names=["facebook"], label_set="full")
    adapter = NormalizedLabelAdapter()

    row = store.get("facebook", "b")
    assert row is not None
    encoded = adapter.encode_row(row)
    assert encoded["class_ids"]["target_presence"] == adapter.vocab.label_to_id("target_presence", "unknown")
    assert encoded["masks"]["target_presence"] == 0
    assert encoded["masks"]["tactic_rhetorical"] == 0
    assert encoded["sample_weight"] == 0.33


def test_ambiguous_auxiliary_labels_have_ignore_masks():
    adapter = NormalizedLabelAdapter()

    assert adapter.vocab.mask_for_single("target_presence", "ambiguous") == 0
    assert adapter.vocab.mask_for_single("target_granularity", "ambiguous") == 0
    assert adapter.vocab.mask_for_single("tactic_multimodal_relation", "ambiguous") == 0


def test_normalized_meme_dataset_attaches_targets_and_metadata(tmp_path):
    source, annotation_root, normalized_root = _write_mini_normalized_dataset(tmp_path)
    dataset = NormalizedMemeDataset(
        dataset_root=source,
        annotation_root=annotation_root,
        normalized_root=normalized_root,
        dataset_names=["facebook"],
        label_set="full",
        keep_missing_images=True,
    )

    assert len(dataset) == 2
    sample = dataset[0]
    assert sample["sample_id"] == "a"
    assert sample["normalized_annotation"] is not None
    assert sample["targets"]["class_ids"]["harmfulness"] == 1
    assert sample["label_strings"]["intent_primary"] == "ridicule_mockery"
    assert sample["evidence_text"]["key_text_evidence"] == "mocking text"
    assert sample["audit_flags"] == ["background_knowledge_needed"]


def test_clean_label_set_reads_normalized_clean(tmp_path):
    source, annotation_root, normalized_root = _write_mini_normalized_dataset(tmp_path)
    dataset = NormalizedMemeDataset(
        dataset_root=source,
        annotation_root=annotation_root,
        normalized_root=normalized_root,
        dataset_names=["facebook"],
        label_set="clean",
        keep_missing_images=True,
    )

    assert len(dataset) == 1
    assert dataset[0]["sample_id"] == "a"


def test_inspect_normalized_labels_cli_runs(tmp_path):
    source, annotation_root, normalized_root = _write_mini_normalized_dataset(tmp_path)
    cmd = [
        sys.executable,
        "scripts/inspect_normalized_labels.py",
        "--dataset",
        "facebook",
        "--label-set",
        "full",
        "--dataset-root",
        str(source),
        "--annotation-root",
        str(annotation_root),
        "--normalized-root",
        str(normalized_root),
        "--limit",
        "2",
    ]
    result = subprocess.run(cmd, cwd=Path.cwd(), check=True, capture_output=True, text=True)
    assert "loaded_samples" in result.stdout
    assert "vocab_sizes" in result.stdout


def _write_mini_normalized_dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source"
    dataset_root = source / "facebook_img+text"
    (dataset_root / "txt").mkdir(parents=True)
    (dataset_root / "img").mkdir(parents=True)
    records = [
        {"id": "a", "image": "a.png", "labels": 1, "text": "mocking text"},
        {"id": "b", "image": "b.png", "labels": 0, "text": "uncertain text"},
    ]
    for record in records:
        (dataset_root / "img" / record["image"]).write_bytes(b"not-real-image")
    (dataset_root / "txt" / "all.jsonl").write_text(
        "\n".join(json.dumps(row) for row in records) + "\n",
        encoding="utf-8",
    )
    annotation_root = tmp_path / "annotation" / "facebook"
    annotation_root.mkdir(parents=True)
    (annotation_root / "facebook_annotations.jsonl").write_text("", encoding="utf-8")

    normalized_root = tmp_path / "annotation_normalized" / "facebook"
    normalized_root.mkdir(parents=True)
    rows = [
        _normalized_row(
            "a",
            harmfulness="harmful",
            target_presence="implicit",
            tactic_rhetorical=["stereotype", "sarcasm_irony"],
            confidence_score=1.0,
            not_sure=False,
            audit_flags=["background_knowledge_needed"],
        ),
        _normalized_row(
            "b",
            harmfulness="non_harmful",
            target_presence="unmapped_target",
            tactic_rhetorical=["unknown"],
            confidence_score=0.66,
            not_sure=True,
            audit_flags=[],
        ),
    ]
    (normalized_root / "normalized_labels.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    (normalized_root / "normalized_clean.jsonl").write_text(
        json.dumps(rows[0]) + "\n",
        encoding="utf-8",
    )
    assert read_jsonl(normalized_root / "normalized_labels.jsonl")
    return source, tmp_path / "annotation", tmp_path / "annotation_normalized"


def _normalized_row(
    sample_id: str,
    *,
    harmfulness: str,
    target_presence: str,
    tactic_rhetorical: list[str],
    confidence_score: float,
    not_sure: bool,
    audit_flags: list[str],
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "dataset_name": "facebook",
        "image_path": f"fake/{sample_id}.png",
        "ocr_text_full": "mocking text",
        "raw_label": 1 if harmfulness == "harmful" else 0,
        "labels": {
            "harmfulness": harmfulness,
            "target_presence": target_presence,
            "target_granularity": "community",
            "protected_attribute": ["religion"],
            "intent_primary": "ridicule_mockery",
            "secondary_intent": ["criticism"],
            "stance": "hostile",
            "background_knowledge_needed": True,
            "tactic_rhetorical": tactic_rhetorical,
            "tactic_multimodal_relation": "cross_modal_implication",
            "not_sure": not_sure,
            "confidence": "medium",
            "confidence_score": confidence_score,
        },
        "evidence_text": {
            "key_text_evidence": "mocking text",
            "key_visual_evidence": "",
            "key_cross_modal_evidence": "",
        },
        "source_annotation": {"has_annotation": True, "annotation_schema_version": "v1_silver"},
        "audit_flags": audit_flags,
    }
