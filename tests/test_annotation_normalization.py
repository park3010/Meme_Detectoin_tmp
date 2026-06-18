from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from experiments.annotation_audit import compute_annotation_audit
from experiments.annotation_normalization import (
    build_normalized_dataset,
    is_clean_row,
    load_normalization_config,
    normalize_sample_annotation,
    write_normalized_outputs,
)
from utils.io import read_jsonl


def test_normal_annotation_and_aliases_are_normalized(tmp_path):
    source, annotation_root = _write_mini_dataset(tmp_path)
    cfg = load_normalization_config("configs/annotation_normalization.yaml")
    rows, warnings = build_normalized_dataset(["facebook"], source, annotation_root, cfg, disable_tqdm=True)

    row = next(item for item in rows if item["sample_id"] == "a")
    assert row["labels"]["harmfulness"] == "non_harmful"
    assert row["labels"]["target_presence"] == "explicit"
    assert row["labels"]["stance"] == "hostile"
    assert row["labels"]["intent_primary"] == "ridicule_mockery"
    assert "sarcasm_irony" in row["labels"]["tactic_rhetorical"]
    assert "raw_label_non_harmful_but_hostile" in row["audit_flags"]
    assert any(w["type"] == "missing_annotation" and w["sample_id"] == "c" for w in warnings)


def test_missing_fields_do_not_crash_and_warn():
    cfg = load_normalization_config("configs/annotation_normalization.yaml")
    sample = {
        "sample_id": "missing",
        "dataset_name": "facebook",
        "image_path": None,
        "ocr_text_full": "",
        "raw_label": "unexpected",
        "annotation": {"annotation": {"quality": {"confidence": "low", "not_sure": "yes"}}},
    }
    row, warnings = normalize_sample_annotation(sample, cfg)
    assert row is not None
    assert row["labels"]["harmfulness"] == "unknown"
    assert row["labels"]["target_presence"] == "unknown"
    assert row["labels"]["not_sure"] is True
    assert "unknown_harmfulness" in row["audit_flags"]
    assert any(w["type"] == "unknown_harmfulness" for w in warnings)


def test_not_sure_is_excluded_from_clean_output_by_default(tmp_path):
    source, annotation_root = _write_mini_dataset(tmp_path)
    cfg = load_normalization_config("configs/annotation_normalization.yaml")
    rows, warnings = build_normalized_dataset(["facebook"], source, annotation_root, cfg, disable_tqdm=True)
    out = write_normalized_outputs(rows, warnings, tmp_path / "normalized")

    clean_rows = read_jsonl(out["facebook_clean"])
    assert {row["sample_id"] for row in clean_rows} == {"a"}
    unsure = next(row for row in rows if row["sample_id"] == "b")
    assert is_clean_row(unsure, {"high", "medium"}) is False


def test_audit_flags_and_distributions_on_normalized_rows(tmp_path):
    source, annotation_root = _write_mini_dataset(tmp_path)
    cfg = load_normalization_config("configs/annotation_normalization.yaml")
    rows, warnings = build_normalized_dataset(["facebook"], source, annotation_root, cfg, disable_tqdm=True)
    audit = compute_annotation_audit(rows, warnings)

    assert audit["summary"]["samples_with_annotation"] == 2
    assert audit["summary"]["samples_without_annotation"] == 1
    assert audit["label_distributions"]["stance"]["hostile"] == 1
    assert audit["audit_flag_counts"]["raw_label_non_harmful_but_hostile"] == 1


def test_annotation_cli_scripts_run_with_limit(tmp_path):
    source, annotation_root = _write_mini_dataset(tmp_path)
    normalized_root = tmp_path / "normalized"
    audit_root = tmp_path / "audit"

    build_cmd = [
        sys.executable,
        "scripts/build_normalized_labels.py",
        "--dataset",
        "facebook",
        "--dataset-root",
        str(source),
        "--annotation-root",
        str(annotation_root),
        "--normalized-root",
        str(normalized_root),
        "--limit",
        "2",
        "--disable-tqdm",
    ]
    audit_cmd = [
        sys.executable,
        "scripts/audit_annotations.py",
        "--dataset",
        "facebook",
        "--dataset-root",
        str(source),
        "--annotation-root",
        str(annotation_root),
        "--audit-root",
        str(audit_root),
        "--limit",
        "2",
        "--disable-tqdm",
    ]
    subprocess.run(build_cmd, cwd=Path.cwd(), check=True)
    subprocess.run(audit_cmd, cwd=Path.cwd(), check=True)

    assert (normalized_root / "facebook" / "normalized_labels.jsonl").exists()
    assert (audit_root / "facebook" / "audit_summary.json").exists()
    assert len(read_jsonl(normalized_root / "facebook" / "normalized_labels.jsonl")) == 2


def _write_mini_dataset(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    dataset_root = source / "facebook_img+text"
    (dataset_root / "txt").mkdir(parents=True)
    (dataset_root / "img").mkdir(parents=True)
    records = [
        {"id": "a", "image": "a.png", "labels": 0, "text": "mocking text"},
        {"id": "b", "image": "b.png", "labels": 1, "text": "needs context"},
        {"id": "c", "image": "c.png", "labels": 1, "text": "no annotation"},
    ]
    for record in records:
        (dataset_root / "img" / record["image"]).write_bytes(b"not-real-image")
    (dataset_root / "txt" / "all.jsonl").write_text(
        "\n".join(json.dumps(row) for row in records) + "\n",
        encoding="utf-8",
    )

    annotation_root = tmp_path / "annotation" / "facebook"
    annotation_root.mkdir(parents=True)
    annotations = [
        {
            "sample_id": "a",
            "dataset_name": "facebook",
            "annotation": {
                "surface_content": {"ocr_text_full": "mocking text"},
                "target": {
                    "target_presence": "present",
                    "target_granularity": "group",
                    "protected_attribute": "religion, nationality",
                    "target_text_span": "mocking",
                },
                "intent": {
                    "intent_primary": "ridicule",
                    "stance": "attack",
                    "background_knowledge_needed": "no",
                    "intent_free_text": "The meme ridicules a group.",
                },
                "tactic": {
                    "tactic_rhetorical": "irony, stereotype",
                    "tactic_multimodal_relation": "mismatch",
                    "evidence_text_span": "mocking",
                },
                "evidence": {"key_text_evidence": "mocking text"},
                "quality": {"confidence": "high", "not_sure": "no"},
            },
        },
        {
            "sample_id": "b",
            "dataset_name": "facebook",
            "annotation": {
                "intent": {"background_knowledge_needed": "yes"},
                "quality": {"confidence": "medium", "not_sure": "yes"},
            },
        },
    ]
    (annotation_root / "facebook_annotations.jsonl").write_text(
        "\n".join(json.dumps(row) for row in annotations) + "\n",
        encoding="utf-8",
    )
    return source, tmp_path / "annotation"
