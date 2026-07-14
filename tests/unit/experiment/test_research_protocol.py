from __future__ import annotations

from dataset.meme_dataset import PAPER_DATASET_PROTOCOL
import pytest

from experiments.research_protocol import _write_immutable_manifest, audit_fhm_leakage, build_fhm_manifest, build_source_manifest, validate_fhm_manifest, validate_source_manifest


def _row(dataset: str, index: int, label: int, eligible: bool = True):
    return {
        "sample_key": f"{dataset}::{index}",
        "sample_id": str(index),
        "dataset_name": dataset,
        "dataset_family": "harmeme" if dataset != "facebook" else "fhm",
        "original_dataset": dataset,
        "domain": "covid" if dataset == "harm_c" else "politics" if dataset == "harm_p" else "social_media",
        "domain_role": "source_train_validation" if dataset != "facebook" else "heldout_target_test",
        "harmfulness": "harmful" if label else "non_harmful",
        "harmfulness_id": label,
        "normalized_label_exists": True,
        "structured_label_eligible": eligible,
    }


def test_source_manifest_is_deterministic_stratified_and_disjoint():
    rows = [_row(dataset, index, index % 2, eligible=index % 3 != 0) for dataset in ("harm_c", "harm_p") for index in range(20)]
    first = build_source_manifest(rows, split_seed=42, train_ratio=0.8)
    second = build_source_manifest(rows, split_seed=42, train_ratio=0.8)
    assert first == second
    assert validate_source_manifest(first)["passed"] is True
    train = {row["sample_key"] for row in first["train"]}
    valid = {row["sample_key"] for row in first["validation"]}
    assert not train & valid
    assert len(train | valid) == len(rows)


def test_fhm_manifest_is_test_only_and_records_silver_provenance():
    manifest = build_fhm_manifest([_row("facebook", index, index % 2) for index in range(6)])
    assert validate_fhm_manifest(manifest)["passed"] is True
    assert "train" not in manifest and "validation" not in manifest
    assert manifest["structured_evaluation_provenance"] == "agent_silver_structured_evaluation"


def test_dataset_family_roles_disable_memotion():
    assert PAPER_DATASET_PROTOCOL["harm_c"]["domain"] == "covid"
    assert PAPER_DATASET_PROTOCOL["harm_p"]["domain"] == "politics"
    assert PAPER_DATASET_PROTOCOL["facebook"]["domain_role"] == "heldout_target_test"
    assert PAPER_DATASET_PROTOCOL["memotion"]["enabled_for_paper"] is False


def test_leakage_audit_blocks_facebook_retrieval_provenance(tmp_path):
    source = build_source_manifest([_row("harm_c", i, i % 2) for i in range(6)])
    fhm = build_fhm_manifest([_row("facebook", i, i % 2) for i in range(4)])
    corpus = tmp_path / "cache" / "corpus.jsonl"
    corpus.parent.mkdir(parents=True)
    corpus.write_text('{"text":"general"}\n', encoding="utf-8")
    (tmp_path / "wiki_manifest.json").write_text('{"datasets":["facebook"]}', encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(f"paths:\n  retrieval_corpus_paths:\n    - {corpus}\n", encoding="utf-8")
    audit = audit_fhm_leakage(source, fhm, config_path=config, registry={"suites": {}})
    assert audit["passed"] is False
    assert audit["errors"][0]["code"] == "fhm_retrieval_leakage"


def test_immutable_manifest_refuses_silent_replacement(tmp_path):
    path = tmp_path / "split.json"
    _write_immutable_manifest(path, {"split": ["a"]}, force=False)
    assert path.with_suffix(".json.sha256").exists()
    with pytest.raises(FileExistsError):
        _write_immutable_manifest(path, {"split": ["b"]}, force=False)
