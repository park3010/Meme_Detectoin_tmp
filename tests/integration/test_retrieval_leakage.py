from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.retrieval_corpus import (
    RetrievalProtocolError,
    audit_retrieval_profile,
    audit_retrieval_root,
    build_retrieval_corpus,
    resolve_retrieval_runtime,
    retrieval_run_manifest_fields,
)
from module.backbone.retrieval import LocalRetrieverAdapter
from utils.io import load_yaml


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def retrieval_fixture(tmp_path: Path) -> dict[str, Path]:
    source_manifest = tmp_path / "source_split.json"
    _write_json(
        source_manifest,
        {
            "schema_version": "harmeme_source_split_v1",
            "protocol": "harmeme_to_fhm_v1",
            "train": [
                {"sample_key": "harm_c::c1", "sample_id": "c1", "original_dataset": "harm_c", "domain": "covid"},
                {"sample_key": "harm_p::p1", "sample_id": "p1", "original_dataset": "harm_p", "domain": "politics"},
            ],
            "validation": [
                {"sample_key": "harm_c::cv", "sample_id": "cv", "original_dataset": "harm_c", "domain": "covid"}
            ],
        },
    )
    covid_cache = tmp_path / "covid.jsonl"
    political_cache = tmp_path / "political.jsonl"
    _write_jsonl(
        covid_cache,
        [
            {"id": "c1", "labels": 1, "query": "alpha query", "topk": [{"kid": "d1", "score": 0.9, "text": "alpha document"}]},
            {"id": "cv", "labels": 0, "query": "validation query", "topk": [{"kid": "d3", "score": 0.8, "text": "validation only"}]},
        ],
    )
    _write_jsonl(
        political_cache,
        [{"id": "p1", "labels": 0, "query": "beta query", "topk": [{"kid": "d1", "score": 0.7, "text": "alpha document"}, {"kid": "d2", "score": 0.6, "text": "beta document"}]}],
    )
    snapshot = tmp_path / "wiki.jsonl"
    _write_jsonl(
        snapshot,
        [
            {"kid": "d1", "text": "alpha document", "title": "Alpha", "source": "wikipedia", "url": "https://example.test/a"},
            {"kid": "d2", "text": "beta document", "title": "Beta", "source": "wikipedia", "url": "https://example.test/b"},
            {"kid": "d3", "text": "validation only", "title": "Validation", "source": "wikipedia", "url": "https://example.test/v"},
        ],
    )
    output = tmp_path / "retrieval" / "harmeme_train_v1"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
paths:
  retrieval_corpus_paths: [{output / 'corpus/corpus_texts.jsonl'}]
retrieval:
  active_profile: harmeme_train_v1
  require_paper_eligible: true
  profiles:
    harmeme_train_v1:
      enabled: true
      paper_eligible: true
      corpus_role: harmeme_train_conditioned
      corpus_root: {output}
      source_split_manifest: {source_manifest}
      source_document_snapshot: {snapshot}
      read_only: true
      allowed_source_partition: train
      replay_sources:
        - dataset_name: harm_c
          domain: covid
          query_cache: {covid_cache}
        - dataset_name: harm_p
          domain: politics
          query_cache: {political_cache}
    legacy_wiki_common:
      enabled: false
      paper_eligible: false
      corpus_role: legacy_non_paper
      corpus_root: {tmp_path / 'wiki_common'}
""",
        encoding="utf-8",
    )
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        f"protocol:\n  source_split_manifest: {source_manifest}\n",
        encoding="utf-8",
    )
    return {
        "source_manifest": source_manifest,
        "covid_cache": covid_cache,
        "political_cache": political_cache,
        "snapshot": snapshot,
        "output": output,
        "config": config,
        "registry": registry,
    }


def _build(paths: dict[str, Path], output: Path | None = None) -> dict:
    return build_retrieval_corpus(
        registry_path=paths["registry"],
        config_path=paths["config"],
        output_root=output or paths["output"],
        offline=True,
    )


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _tamper_first_row(path: Path, update) -> None:
    rows = _rows(path)
    update(rows[0])
    _write_jsonl(path, rows)


def test_harmeme_train_ids_are_accepted(retrieval_fixture):
    result = _build(retrieval_fixture)
    assert result["query_origin_sample_count"] == 2
    assert result["document_count"] == 2


def test_harmeme_validation_ids_are_rejected(retrieval_fixture):
    _build(retrieval_fixture)
    query_rows = _rows(retrieval_fixture["output"] / "cache/source_queries/queries.jsonl")
    assert {row["sample_key"] for row in query_rows} == {"harm_c::c1", "harm_p::p1"}
    assert "d3" not in {row["document_id"] for row in _rows(retrieval_fixture["output"] / "corpus/corpus_texts.jsonl")}


def test_facebook_origin_is_rejected(retrieval_fixture):
    _build(retrieval_fixture)
    corpus = retrieval_fixture["output"] / "corpus/corpus_texts.jsonl"
    _tamper_first_row(corpus, lambda row: row["origin_sample_ids"].append("facebook::f1"))
    audit = audit_retrieval_root(retrieval_fixture["output"], expected_source_manifest=retrieval_fixture["source_manifest"])
    assert "fhm_retrieval_leakage" in {error["code"] for error in audit["errors"]}


def test_memotion_origin_is_rejected(retrieval_fixture):
    _build(retrieval_fixture)
    corpus = retrieval_fixture["output"] / "corpus/corpus_texts.jsonl"
    _tamper_first_row(corpus, lambda row: row["origin_sample_ids"].append("memotion::m1"))
    audit = audit_retrieval_root(retrieval_fixture["output"], expected_source_manifest=retrieval_fixture["source_manifest"])
    assert "disabled_dataset_retrieval_leakage" in {error["code"] for error in audit["errors"]}


def test_unknown_origin_is_rejected(retrieval_fixture):
    _build(retrieval_fixture)
    corpus = retrieval_fixture["output"] / "corpus/corpus_texts.jsonl"
    _tamper_first_row(corpus, lambda row: row["origin_sample_ids"].append("harm_c::unknown"))
    audit = audit_retrieval_root(retrieval_fixture["output"], expected_source_manifest=retrieval_fixture["source_manifest"])
    assert "retrieval_unknown_origin" in {error["code"] for error in audit["errors"]}


def test_split_manifest_sha_mismatch_is_rejected(retrieval_fixture):
    _build(retrieval_fixture)
    source = json.loads(retrieval_fixture["source_manifest"].read_text())
    source["note"] = "changed"
    _write_json(retrieval_fixture["source_manifest"], source)
    audit = audit_retrieval_root(retrieval_fixture["output"], expected_source_manifest=retrieval_fixture["source_manifest"])
    assert "retrieval_source_split_hash_mismatch" in {error["code"] for error in audit["errors"]}


def test_corpus_hash_mismatch_is_rejected(retrieval_fixture):
    _build(retrieval_fixture)
    corpus = retrieval_fixture["output"] / "corpus/corpus_texts.jsonl"
    corpus.write_text(corpus.read_text() + "\n", encoding="utf-8")
    audit = audit_retrieval_root(retrieval_fixture["output"], expected_source_manifest=retrieval_fixture["source_manifest"])
    assert "retrieval_corpus_hash_mismatch" in {error["code"] for error in audit["errors"]}


def test_index_hash_mismatch_is_rejected(retrieval_fixture):
    _build(retrieval_fixture)
    dense = retrieval_fixture["output"] / "index/dense/embeddings.f32"
    dense.write_bytes(dense.read_bytes() + b"x")
    audit = audit_retrieval_root(retrieval_fixture["output"], expected_source_manifest=retrieval_fixture["source_manifest"])
    assert "retrieval_index_hash_mismatch" in {error["code"] for error in audit["errors"]}


def test_legacy_wiki_common_cannot_pass_paper_audit(retrieval_fixture):
    legacy = retrieval_fixture["output"].parent / "wiki_common"
    legacy.mkdir(parents=True)
    audit = audit_retrieval_root(legacy, expected_profile="legacy_wiki_common", expected_source_manifest=retrieval_fixture["source_manifest"])
    assert not audit["passed"]
    assert "retrieval_legacy_profile_selected" in {error["code"] for error in audit["errors"]}


def test_fhm_may_query_frozen_valid_corpus(retrieval_fixture, tmp_path):
    _build(retrieval_fixture)
    root = retrieval_fixture["output"]
    before = (_sha(root / "corpus/corpus_texts.jsonl"), _sha(root / "index_manifest.json"))
    adapter = LocalRetrieverAdapter(
        [root / "corpus/corpus_texts.jsonl"],
        fallback_candidates=False,
        index_root=root / "index",
        query_cache_root=tmp_path / "run/retrieval_queries",
    )
    results = adapter.search("alpha", query_context={"dataset_name": "facebook", "sample_id": "f1"})
    assert results
    assert before == (_sha(root / "corpus/corpus_texts.jsonl"), _sha(root / "index_manifest.json"))


def test_fhm_query_results_are_run_scoped_and_label_free(retrieval_fixture, tmp_path):
    _build(retrieval_fixture)
    run_root = tmp_path / "run/retrieval_queries"
    adapter = LocalRetrieverAdapter(
        [retrieval_fixture["output"] / "corpus/corpus_texts.jsonl"],
        query_cache_root=run_root,
    )
    adapter.search("alpha", query_context={"dataset_name": "facebook", "sample_id": "f1"})
    records = _rows(run_root / "fhm/queries.jsonl")
    assert records[0]["contains_labels"] is False
    assert not ({"label", "labels", "gold_label"} & records[0].keys())
    assert not (retrieval_fixture["output"] / "retrieval_queries").exists()


def test_build_is_deterministic(retrieval_fixture, tmp_path):
    first = _build(retrieval_fixture)
    second = _build(retrieval_fixture, tmp_path / "second")
    assert first["corpus_sha256"] == second["corpus_sha256"]
    assert first["sparse_index_sha256"] == second["sparse_index_sha256"]
    assert first["dense_index_sha256"] == second["dense_index_sha256"]


def test_existing_valid_output_is_not_rebuilt(retrieval_fixture):
    _build(retrieval_fixture)
    second = _build(retrieval_fixture)
    assert second["action"] == "already_valid"


def test_existing_different_output_requires_force(retrieval_fixture):
    _build(retrieval_fixture)
    with pytest.raises(RetrievalProtocolError, match="retrieval_output_exists"):
        build_retrieval_corpus(
            registry_path=retrieval_fixture["registry"],
            config_path=retrieval_fixture["config"],
            output_root=retrieval_fixture["output"],
            limit=1,
        )


def test_atomic_failure_leaves_no_partial_output(retrieval_fixture, tmp_path):
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="synthetic failure"):
        build_retrieval_corpus(
            registry_path=retrieval_fixture["registry"],
            config_path=retrieval_fixture["config"],
            output_root=output,
            _fail_after="corpus",
        )
    assert not output.exists()
    assert not list(output.parent.glob(".failed.tmp-*"))


def test_row_provenance_survives_deduplication(retrieval_fixture):
    _build(retrieval_fixture)
    row = next(row for row in _rows(retrieval_fixture["output"] / "corpus/corpus_texts.jsonl") if row["document_id"] == "d1")
    assert row["origin_sample_ids"] == ["harm_c::c1", "harm_p::p1"]
    assert row["origin_original_datasets"] == ["harm_c", "harm_p"]
    assert len(row["origin_query_ids"]) == 2


def test_query_origins_are_subset_of_immutable_train(retrieval_fixture):
    _build(retrieval_fixture)
    source = json.loads(retrieval_fixture["source_manifest"].read_text())
    train = {row["sample_key"] for row in source["train"]}
    query = {row["sample_key"] for row in _rows(retrieval_fixture["output"] / "cache/source_queries/queries.jsonl")}
    assert query <= train


def test_run_manifest_fields_record_corpus_and_index_hashes(retrieval_fixture, tmp_path):
    _build(retrieval_fixture)
    runtime = resolve_retrieval_runtime(json.loads("{}"), disabled=True)
    assert retrieval_run_manifest_fields(runtime)["retrieval_read_only"] is True
    config = load_yaml(retrieval_fixture["config"])
    runtime = resolve_retrieval_runtime(config)
    fields = retrieval_run_manifest_fields(runtime, tmp_path / "run/retrieval_queries")
    assert fields["retrieval_corpus_sha256"]
    assert fields["retrieval_sparse_index_sha256"]
    assert fields["retrieval_dense_index_sha256"]


def test_without_retrieval_loads_no_legacy_corpus(retrieval_fixture):
    config = load_yaml(retrieval_fixture["config"])
    config["paths"]["retrieval_corpus_paths"] = ["dataset/source/wiki_common/cache/corpus_texts.jsonl"]
    runtime = resolve_retrieval_runtime(config, disabled=True)
    assert runtime["corpus_paths"] == []
    assert runtime["reason"] == "retrieval_disabled_by_experiment_contract"


def test_strict_profile_audit_passes_for_valid_build(retrieval_fixture):
    _build(retrieval_fixture)
    audit = audit_retrieval_profile(
        registry_path=retrieval_fixture["registry"],
        config_path=retrieval_fixture["config"],
        strict=True,
    )
    assert audit["passed"]
    assert all(audit["checks"].values())


def test_strict_profile_audit_fails_for_legacy_selection(retrieval_fixture):
    config = retrieval_fixture["config"].read_text(encoding="utf-8").replace(
        "active_profile: harmeme_train_v1", "active_profile: legacy_wiki_common"
    )
    retrieval_fixture["config"].write_text(config, encoding="utf-8")
    audit = audit_retrieval_profile(
        registry_path=retrieval_fixture["registry"],
        config_path=retrieval_fixture["config"],
        profile="legacy_wiki_common",
        strict=True,
    )
    assert not audit["passed"]
    assert "retrieval_legacy_profile_selected" in {error["code"] for error in audit["errors"]}


def test_limited_build_is_not_paper_eligible(retrieval_fixture, tmp_path):
    result = build_retrieval_corpus(
        registry_path=retrieval_fixture["registry"],
        config_path=retrieval_fixture["config"],
        output_root=tmp_path / "limited",
        limit=1,
    )
    assert result["paper_eligible"] is False


def test_builder_refuses_network_mode(retrieval_fixture):
    with pytest.raises(RetrievalProtocolError, match="retrieval_network_access_forbidden"):
        build_retrieval_corpus(
            registry_path=retrieval_fixture["registry"],
            config_path=retrieval_fixture["config"],
            output_root=retrieval_fixture["output"],
            offline=False,
        )
