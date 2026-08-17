"""Leakage-safe retrieval corpus construction, validation, and resolution.

The canonical paper corpus is replayed from auditable per-query caches. The
aggregate legacy corpus is never filtered or copied as a corpus/index unit.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from experiments.research_protocol import sha256_file
from utils.io import load_yaml
from utils.tensor_utils import hashed_vector


MANIFEST_SCHEMA = "retrieval_corpus_manifest_v2"
INDEX_MANIFEST_SCHEMA = "retrieval_index_manifest_v1"
PROTOCOL_NAME = "harmeme_to_fhm_v1"
CANONICAL_PROFILE = "harmeme_train_v1"
CANONICAL_ROLE = "harmeme_train_conditioned"
STATIC_ROLE = "static_external_dataset_independent"
LEGACY_PROFILE = "legacy_wiki_common"
ALLOWED_DATASETS = {"harm_c", "harm_p"}
FORBIDDEN_DATASETS = {"facebook", "fhm", "memotion"}
DATASET_DOMAINS = {"harm_c": "covid", "harm_p": "politics"}
DEFAULT_SOURCE_MANIFEST = "result/splits/harmeme/source_split_seed_42.json"
DEFAULT_OUTPUT_ROOT = "dataset/retrieval/harmeme_train_v1"
QUERY_TYPE = "legacy_combined_query"
DENSE_DIM = 256

REQUIRED_ROW_FIELDS = {
    "document_id",
    "text",
    "title",
    "source_name",
    "source_uri_or_identifier",
    "retrieval_source_type",
    "origin_sample_ids",
    "origin_original_datasets",
    "origin_domains",
    "origin_split",
    "origin_query_ids",
    "origin_query_types",
    "is_dataset_internal_meme",
    "is_external_document",
    "created_from_harmeme_train_only",
}


class RetrievalProtocolError(RuntimeError):
    """Protocol failure carrying a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ReplaySource:
    """One eligible source-dataset query cache."""

    dataset_name: str
    domain: str
    query_cache: Path


def build_retrieval_corpus(
    *,
    registry_path: str | Path = "configs/experiment_registry.yaml",
    config_path: str | Path = "configs/config.yaml",
    profile: str = CANONICAL_PROFILE,
    source_partition: str = "train",
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    offline: bool = True,
    force: bool = False,
    limit: int | None = None,
    _fail_after: str | None = None,
) -> dict[str, Any]:
    """Replay a canonical corpus from immutable-train query caches.

    The private ``_fail_after`` hook exists only for atomicity tests. Online
    collection is intentionally unsupported in this remediation.
    """

    if not offline:
        raise RetrievalProtocolError(
            "retrieval_network_access_forbidden",
            "This builder is offline-only; live source collection requires a separately reviewed protocol.",
        )
    if profile != CANONICAL_PROFILE:
        raise RetrievalProtocolError(
            "retrieval_profile_invalid",
            f"The canonical builder only creates {CANONICAL_PROFILE!r}, not {profile!r}.",
        )
    if source_partition != "train":
        raise RetrievalProtocolError(
            "retrieval_source_partition_invalid",
            "The paper corpus may be built only from the immutable HarMeme train partition.",
        )
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided")

    config = load_yaml(config_path)
    registry = load_yaml(registry_path)
    profile_cfg = retrieval_profile_config(config, profile)
    configured_root = Path(str(profile_cfg.get("corpus_root", DEFAULT_OUTPUT_ROOT)))
    destination = Path(output_root)
    if destination == Path(DEFAULT_OUTPUT_ROOT) and configured_root != Path(DEFAULT_OUTPUT_ROOT):
        destination = configured_root

    source_manifest_path = Path(
        str(
            profile_cfg.get("source_split_manifest")
            or (registry.get("protocol", {}) or {}).get("source_split_manifest")
            or DEFAULT_SOURCE_MANIFEST
        )
    )
    if not source_manifest_path.exists():
        raise RetrievalProtocolError(
            "blocked_retrieval_corpus_rebuild",
            f"Immutable source split manifest is missing: {source_manifest_path}",
        )
    source_manifest = _read_json(source_manifest_path)
    source_sha = sha256_file(source_manifest_path)
    train_rows = list(source_manifest.get("train", []) or [])
    if not train_rows:
        raise RetrievalProtocolError(
            "blocked_retrieval_corpus_rebuild", "Immutable source manifest has no train rows."
        )
    _validate_source_rows(train_rows)
    selected_rows = train_rows[:limit] if limit is not None else train_rows
    allowed_rows = {str(row["sample_key"]): row for row in selected_rows}
    replay_sources = _replay_sources(profile_cfg)
    source_snapshot = Path(str(profile_cfg.get("source_document_snapshot", "")))
    if not source_snapshot.is_file():
        raise RetrievalProtocolError(
            "blocked_retrieval_corpus_rebuild",
            f"Local source-document snapshot is missing: {source_snapshot}",
        )

    request = {
        "profile": profile,
        "source_partition": source_partition,
        "source_split_manifest_sha256": source_sha,
        "limit": limit,
    }
    if destination.exists():
        current = audit_retrieval_root(
            destination,
            expected_profile=profile,
            expected_source_manifest=source_manifest_path,
            strict=limit is None,
        )
        current_request = (current.get("manifest") or {}).get("build_request", {})
        if current.get("passed") and current_request == request:
            return {**_build_summary(destination, current), "action": "already_valid"}
        if not force:
            raise RetrievalProtocolError(
                "retrieval_output_exists",
                f"Output exists and does not match this build request: {destination}; use --force to replace atomically.",
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        result = _build_into(
            temporary,
            source_manifest_path=source_manifest_path,
            source_manifest=source_manifest,
            source_sha=source_sha,
            train_rows=train_rows,
            allowed_rows=allowed_rows,
            replay_sources=replay_sources,
            source_snapshot=source_snapshot,
            config_path=Path(config_path),
            profile_cfg=profile_cfg,
            request=request,
            limit=limit,
            fail_after=_fail_after,
        )
        audit = audit_retrieval_root(
            temporary,
            expected_profile=profile,
            expected_source_manifest=source_manifest_path,
            strict=limit is None,
        )
        if not audit["passed"]:
            codes = ", ".join(error["code"] for error in audit["errors"])
            raise RetrievalProtocolError(
                "retrieval_manifest_invalid", f"Temporary build failed self-audit: {codes}"
            )
        _atomic_install(temporary, destination, force=force)
        return {**result, "action": "built", "output_root": str(destination)}
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _build_into(
    root: Path,
    *,
    source_manifest_path: Path,
    source_manifest: dict[str, Any],
    source_sha: str,
    train_rows: list[dict[str, Any]],
    allowed_rows: dict[str, dict[str, Any]],
    replay_sources: list[ReplaySource],
    source_snapshot: Path,
    config_path: Path,
    profile_cfg: dict[str, Any],
    request: dict[str, Any],
    limit: int | None,
    fail_after: str | None,
) -> dict[str, Any]:
    corpus_path = root / "corpus" / "corpus_texts.jsonl"
    query_cache_path = root / "cache" / "source_queries" / "queries.jsonl"
    source_docs_path = root / "cache" / "source_documents" / "documents.jsonl"
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    query_cache_path.parent.mkdir(parents=True, exist_ok=True)
    source_docs_path.parent.mkdir(parents=True, exist_ok=True)

    query_rows, document_origins = _select_train_queries(allowed_rows, replay_sources)
    selected_keys = {row["sample_key"] for row in query_rows}
    missing = sorted(set(allowed_rows) - selected_keys)
    if missing:
        raise RetrievalProtocolError(
            "blocked_retrieval_corpus_rebuild",
            f"{len(missing)} selected train samples have no auditable query cache row; examples={missing[:5]}",
        )
    selected_document_ids = set(document_origins)
    source_documents = _load_source_documents(source_snapshot, selected_document_ids)
    missing_docs = sorted(selected_document_ids - set(source_documents))
    if missing_docs:
        raise RetrievalProtocolError(
            "blocked_retrieval_corpus_rebuild",
            f"{len(missing_docs)} cached document IDs are absent from the local source snapshot; examples={missing_docs[:5]}",
        )

    corpus_rows = [
        _corpus_row(document_id, source_documents[document_id], document_origins[document_id])
        for document_id in sorted(document_origins)
    ]
    _write_jsonl(query_cache_path, query_rows)
    _write_jsonl(source_docs_path, [source_documents[key] for key in sorted(source_documents)])
    _write_jsonl(corpus_path, corpus_rows)
    if fail_after == "corpus":
        raise RuntimeError("synthetic failure after corpus write")

    corpus_sha = sha256_file(corpus_path)
    query_ids = sorted(selected_keys)
    query_ids_sha = _sha256_lines(query_ids)
    code_sha = sha256_file(Path(__file__))
    builder_config_sha = _sha256_json(
        {
            "config_sha256": sha256_file(config_path),
            "profile": profile_cfg,
            "query_type": QUERY_TYPE,
            "dense_dim": DENSE_DIM,
        }
    )
    train_total = len(train_rows)
    limited = limit is not None and len(query_rows) < train_total
    created_at = _reproducible_timestamp()
    retrieval_manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "protocol_name": PROTOCOL_NAME,
        "retrieval_profile": CANONICAL_PROFILE,
        "corpus_role": CANONICAL_ROLE,
        "paper_eligible": not limited,
        "source_split_manifest_path": str(source_manifest_path),
        "source_split_manifest_sha256": source_sha,
        "source_split_content_sha256": source_manifest.get("content_sha256"),
        "source_partition": "train",
        "source_train_total_count": train_total,
        "included_original_datasets": ["harm_c", "harm_p"],
        "included_domains": ["covid", "politics"],
        "excluded_original_datasets": ["facebook", "memotion"],
        "excluded_partitions": ["validation", "test"],
        "query_origin_sample_count": len(query_rows),
        "query_origin_sample_coverage": len(query_rows) / train_total,
        "query_origin_sample_ids_sha256": query_ids_sha,
        "query_origin_exclusion_reasons": (
            {"test_only_limit": train_total - len(query_rows)} if limited else {}
        ),
        "query_id_derivation": "sha256(dataset_name + sample_id + exact_cached_query)",
        "source_query_cache_file": "cache/source_queries/queries.jsonl",
        "source_query_cache_sha256": sha256_file(query_cache_path),
        "source_document_cache_file": "cache/source_documents/documents.jsonl",
        "source_document_cache_sha256": sha256_file(source_docs_path),
        "document_count": len(corpus_rows),
        "corpus_file": "corpus/corpus_texts.jsonl",
        "corpus_sha256": corpus_sha,
        "builder_config_sha256": builder_config_sha,
        "code_sha256": code_sha,
        "network_access_used": False,
        "build_mode": "offline_query_cache_replay",
        "created_at_utc": created_at,
        "build_timestamp_policy": "SOURCE_DATE_EPOCH_or_unix_epoch",
        "build_request": request,
        "forbidden_origin_counts": {
            "harmeme_validation": 0,
            "facebook": 0,
            "memotion": 0,
            "unknown": 0,
        },
    }
    retrieval_manifest_path = root / "retrieval_manifest.json"
    _write_json(retrieval_manifest_path, retrieval_manifest)
    retrieval_manifest_sha = sha256_file(retrieval_manifest_path)

    sparse_files, dense_files = _build_indexes(root, corpus_rows)
    if fail_after == "index":
        raise RuntimeError("synthetic failure after index write")
    sparse_sha = _sha256_tree(root / "index" / "sparse")
    dense_sha = _sha256_tree(root / "index" / "dense")
    embedding_spec = f"hashed_vector_v1_dim_{DENSE_DIM}"
    index_manifest = {
        "schema_version": INDEX_MANIFEST_SCHEMA,
        "retrieval_profile": CANONICAL_PROFILE,
        "corpus_manifest_path": "retrieval_manifest.json",
        "corpus_manifest_sha256": retrieval_manifest_sha,
        "corpus_sha256": corpus_sha,
        "sparse_backend": "bm25_token_statistics_v1",
        "dense_backend": "hashed_vector_binary_v1",
        "embedding_model": embedding_spec,
        "embedding_asset_sha256": hashlib.sha256(embedding_spec.encode("utf-8")).hexdigest(),
        "sparse_index_files": sparse_files,
        "dense_index_files": dense_files,
        "sparse_index_sha256": sparse_sha,
        "dense_index_sha256": dense_sha,
        "indexed_document_count": len(corpus_rows),
        "created_at_utc": created_at,
        "code_sha256": code_sha,
        "source_corpus_role": CANONICAL_ROLE,
    }
    index_manifest_path = root / "index_manifest.json"
    _write_json(index_manifest_path, index_manifest)
    _write_checksums(root)
    return {
        "status": "pass",
        "retrieval_profile": CANONICAL_PROFILE,
        "source_split_manifest_sha256": source_sha,
        "query_origin_sample_count": len(query_rows),
        "query_origin_sample_coverage": len(query_rows) / train_total,
        "document_count": len(corpus_rows),
        "corpus_sha256": corpus_sha,
        "sparse_index_sha256": sparse_sha,
        "dense_index_sha256": dense_sha,
        "paper_eligible": not limited,
    }


def audit_retrieval_profile(
    *,
    registry_path: str | Path = "configs/experiment_registry.yaml",
    config_path: str | Path = "configs/config.yaml",
    profile: str | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Audit a configured retrieval profile and immutable split binding."""

    config = load_yaml(config_path)
    registry = load_yaml(registry_path)
    active = str((config.get("retrieval", {}) or {}).get("active_profile") or "")
    selected = profile or active
    errors: list[dict[str, Any]] = []
    if not selected:
        return _audit_failure("retrieval_profile_missing", "No retrieval profile is selected.")
    try:
        profile_cfg = retrieval_profile_config(config, selected)
    except RetrievalProtocolError as exc:
        return _audit_failure(exc.code, exc.message)
    root = Path(str(profile_cfg.get("corpus_root", "")))
    source_manifest = Path(
        str(
            profile_cfg.get("source_split_manifest")
            or (registry.get("protocol", {}) or {}).get("source_split_manifest")
            or DEFAULT_SOURCE_MANIFEST
        )
    )
    audit = audit_retrieval_root(
        root,
        expected_profile=selected,
        expected_source_manifest=source_manifest,
        strict=strict,
    )
    errors.extend(audit["errors"])
    if strict and selected != CANONICAL_PROFILE:
        _append_error(errors, "retrieval_legacy_profile_selected", "Strict paper audit requires harmeme_train_v1.")
    if strict and selected != active:
        _append_error(
            errors,
            "retrieval_profile_missing",
            f"Audited profile {selected!r} is not the runtime active profile {active!r}.",
        )
    if strict and not bool(profile_cfg.get("paper_eligible", False)):
        _append_error(errors, "retrieval_legacy_profile_selected", "Selected profile is not paper eligible.")
    if strict and profile_cfg.get("read_only") is not True:
        _append_error(errors, "retrieval_manifest_invalid", "Paper retrieval profile must be read-only.")
    if strict and profile_cfg.get("allowed_source_partition") != "train":
        _append_error(errors, "retrieval_source_partition_invalid", "Runtime profile must allow only the train partition.")
    checks = dict(audit.get("checks", {}))
    checks["runtime_config_points_to_same_profile"] = selected == active
    checks["retrieval_profile_is_paper_eligible"] = bool(
        profile_cfg.get("paper_eligible", False)
        and (audit.get("manifest") or {}).get("paper_eligible", False)
    )
    passed = not errors
    return {
        **audit,
        "passed": passed,
        "status": "pass" if passed else "fail",
        "strict": strict,
        "active_profile": active,
        "retrieval_profile": selected,
        "profile_config": profile_cfg,
        "checks": checks,
        "errors": errors,
    }


def audit_retrieval_root(
    root: str | Path,
    *,
    expected_profile: str = CANONICAL_PROFILE,
    expected_source_manifest: str | Path | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Validate corpus rows, manifests, hashes, indexes, and source origins."""

    corpus_root = Path(root)
    errors: list[dict[str, Any]] = []
    checks = {
        "fhm_absent_from_retrieval_databases": False,
        "memotion_absent_from_retrieval_databases": False,
        "source_validation_absent_from_retrieval_databases": False,
        "retrieval_index_is_harmeme_train_only": False,
        "retrieval_split_hash_verified": False,
        "retrieval_corpus_hash_verified": False,
        "retrieval_index_hash_verified": False,
        "retrieval_profile_is_paper_eligible": False,
    }
    if "wiki_common" in corpus_root.parts or expected_profile == LEGACY_PROFILE:
        _append_error(errors, "retrieval_legacy_profile_selected", "Legacy mixed-provenance corpus cannot pass paper audit.")
    retrieval_manifest_path = corpus_root / "retrieval_manifest.json"
    index_manifest_path = corpus_root / "index_manifest.json"
    if not retrieval_manifest_path.is_file():
        _append_error(errors, "retrieval_manifest_missing", f"Missing {retrieval_manifest_path}")
        return _root_audit_result(corpus_root, checks, errors)
    try:
        manifest = _read_json(retrieval_manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _append_error(errors, "retrieval_manifest_invalid", str(exc))
        return _root_audit_result(corpus_root, checks, errors)

    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        _append_error(errors, "retrieval_manifest_invalid", "Unsupported retrieval manifest schema.")
    if manifest.get("retrieval_profile") != expected_profile:
        _append_error(errors, "retrieval_manifest_invalid", "Retrieval profile does not match the selected profile.")
    if strict and manifest.get("corpus_role") not in {CANONICAL_ROLE, STATIC_ROLE}:
        _append_error(errors, "retrieval_manifest_invalid", "Corpus role is not valid for paper retrieval.")
    if strict and not manifest.get("paper_eligible"):
        _append_error(errors, "retrieval_manifest_invalid", "Corpus manifest is not paper eligible.")
    checks["retrieval_profile_is_paper_eligible"] = bool(manifest.get("paper_eligible"))
    if manifest.get("source_partition") != "train":
        _append_error(errors, "retrieval_source_partition_invalid", "Source partition must be train.")
    if manifest.get("network_access_used") is not False:
        _append_error(errors, "retrieval_manifest_invalid", "Canonical replay must declare network_access_used=false.")
    if set(manifest.get("included_original_datasets", []) or []) != ALLOWED_DATASETS:
        _append_error(errors, "retrieval_unknown_origin", "Included datasets must be exactly harm_c and harm_p.")
    if FORBIDDEN_DATASETS & {str(value).lower() for value in manifest.get("included_original_datasets", []) or []}:
        _append_error(errors, "retrieval_unknown_origin", "Manifest includes a forbidden dataset.")

    declared_source = Path(str(manifest.get("source_split_manifest_path", "")))
    source_path = Path(expected_source_manifest) if expected_source_manifest is not None else declared_source
    if declared_source and expected_source_manifest is not None and _same_path_text(declared_source, source_path) is False:
        _append_error(errors, "retrieval_source_split_hash_mismatch", "Declared source split path differs from the expected immutable manifest.")
    source_rows: dict[str, dict[str, Any]] = {}
    validation_keys: set[str] = set()
    if source_path.is_file():
        source_sha = sha256_file(source_path)
        checks["retrieval_split_hash_verified"] = source_sha == manifest.get("source_split_manifest_sha256")
        if not checks["retrieval_split_hash_verified"]:
            _append_error(errors, "retrieval_source_split_hash_mismatch", "Source split manifest SHA-256 differs.")
        source_obj = _read_json(source_path)
        source_rows = {str(row.get("sample_key")): row for row in source_obj.get("train", []) or []}
        validation_keys = {str(row.get("sample_key")) for row in source_obj.get("validation", []) or []}
        if int(manifest.get("source_train_total_count", -1)) != len(source_rows):
            _append_error(errors, "retrieval_manifest_invalid", "Source train count differs from immutable manifest.")
    else:
        _append_error(errors, "retrieval_source_split_hash_mismatch", f"Source split manifest is missing: {source_path}")

    corpus_path = corpus_root / str(manifest.get("corpus_file", "corpus/corpus_texts.jsonl"))
    rows: list[dict[str, Any]] = []
    if not corpus_path.is_file():
        _append_error(errors, "retrieval_manifest_invalid", f"Corpus file is missing: {corpus_path}")
    else:
        corpus_sha = sha256_file(corpus_path)
        checks["retrieval_corpus_hash_verified"] = corpus_sha == manifest.get("corpus_sha256")
        if not checks["retrieval_corpus_hash_verified"]:
            _append_error(errors, "retrieval_corpus_hash_mismatch", "Corpus SHA-256 differs from retrieval manifest.")
        try:
            rows = list(_iter_jsonl(corpus_path))
        except ValueError as exc:
            _append_error(errors, "retrieval_manifest_invalid", str(exc))
    if len(rows) != int(manifest.get("document_count", -1)):
        _append_error(errors, "retrieval_manifest_invalid", "Corpus document count differs from manifest.")

    fhm_count = memotion_count = validation_count = unknown_count = 0
    observed_query_origins: set[str] = set()
    for index, row in enumerate(rows):
        missing_fields = sorted(REQUIRED_ROW_FIELDS - set(row))
        if missing_fields:
            _append_error(errors, "retrieval_manifest_invalid", f"Corpus row {index} lacks {missing_fields}.")
            continue
        origins = {str(item) for item in row.get("origin_sample_ids", []) or []}
        datasets = {str(item).lower() for item in row.get("origin_original_datasets", []) or []}
        observed_query_origins.update(origins)
        fhm_count += len(datasets & {"facebook", "fhm"}) + sum(key.startswith("facebook::") for key in origins)
        memotion_count += int("memotion" in datasets) + sum(key.startswith("memotion::") for key in origins)
        validation_count += len(origins & validation_keys)
        unknown_count += len({key for key in origins if key not in source_rows and key not in validation_keys})
        if row.get("origin_split") != "train" or not row.get("created_from_harmeme_train_only"):
            validation_count += 1
        if not origins or not origins.issubset(source_rows):
            unknown_count += int(not origins)
    forbidden = manifest.get("forbidden_origin_counts", {}) or {}
    fhm_count += int(forbidden.get("facebook", 0) or 0)
    memotion_count += int(forbidden.get("memotion", 0) or 0)
    validation_count += int(forbidden.get("harmeme_validation", 0) or 0)
    unknown_count += int(forbidden.get("unknown", 0) or 0)
    checks["fhm_absent_from_retrieval_databases"] = fhm_count == 0
    checks["memotion_absent_from_retrieval_databases"] = memotion_count == 0
    checks["source_validation_absent_from_retrieval_databases"] = validation_count == 0
    if fhm_count:
        _append_error(errors, "fhm_retrieval_leakage", f"Found {fhm_count} FHM origins.")
    if memotion_count:
        _append_error(errors, "disabled_dataset_retrieval_leakage", f"Found {memotion_count} Memotion origins.")
    if validation_count:
        _append_error(errors, "retrieval_validation_leakage", f"Found {validation_count} validation/non-train origins.")
    if unknown_count:
        _append_error(errors, "retrieval_unknown_origin", f"Found {unknown_count} unknown origins.")

    query_cache_path = corpus_root / str(manifest.get("source_query_cache_file", "cache/source_queries/queries.jsonl"))
    query_keys: set[str] = set()
    if query_cache_path.is_file():
        for row in _iter_jsonl(query_cache_path):
            key = str(row.get("sample_key", ""))
            query_keys.add(key)
            if key.startswith(("facebook::", "fhm::")):
                fhm_count += 1
            elif key.startswith("memotion::"):
                memotion_count += 1
            elif key in validation_keys:
                validation_count += 1
            elif key not in source_rows:
                unknown_count += 1
            if "label" in row or "labels" in row or "gold_label" in row:
                _append_error(errors, "retrieval_manifest_invalid", "Canonical query cache contains labels.")
        if sha256_file(query_cache_path) != manifest.get("source_query_cache_sha256"):
            _append_error(errors, "retrieval_manifest_invalid", "Source query cache hash differs.")
    else:
        _append_error(errors, "retrieval_manifest_invalid", "Source query cache is missing.")
    if len(query_keys) != int(manifest.get("query_origin_sample_count", -1)):
        _append_error(errors, "retrieval_manifest_invalid", "Query-origin sample count differs.")
    if _sha256_lines(sorted(query_keys)) != manifest.get("query_origin_sample_ids_sha256"):
        _append_error(errors, "retrieval_manifest_invalid", "Query-origin sample ID hash differs.")
    expected_coverage = len(query_keys) / len(source_rows) if source_rows else 0.0
    if abs(float(manifest.get("query_origin_sample_coverage", -1.0)) - expected_coverage) > 1e-12:
        _append_error(errors, "retrieval_manifest_invalid", "Query-origin sample coverage differs.")
    if not observed_query_origins.issubset(query_keys):
        _append_error(errors, "retrieval_unknown_origin", "Corpus origin is absent from canonical source query cache.")
    source_docs_path = corpus_root / str(
        manifest.get("source_document_cache_file", "cache/source_documents/documents.jsonl")
    )
    if not source_docs_path.is_file() or sha256_file(source_docs_path) != manifest.get("source_document_cache_sha256"):
        _append_error(errors, "retrieval_manifest_invalid", "Source document cache is missing or its hash differs.")
    checks["fhm_absent_from_retrieval_databases"] = fhm_count == 0
    checks["memotion_absent_from_retrieval_databases"] = memotion_count == 0
    checks["source_validation_absent_from_retrieval_databases"] = validation_count == 0
    if fhm_count:
        _append_error(errors, "fhm_retrieval_leakage", f"Found {fhm_count} FHM origins.")
    if memotion_count:
        _append_error(errors, "disabled_dataset_retrieval_leakage", f"Found {memotion_count} Memotion origins.")
    if validation_count:
        _append_error(errors, "retrieval_validation_leakage", f"Found {validation_count} validation/non-train origins.")
    if unknown_count:
        _append_error(errors, "retrieval_unknown_origin", f"Found {unknown_count} unknown origins.")

    if not index_manifest_path.is_file():
        _append_error(errors, "retrieval_index_manifest_missing", f"Missing {index_manifest_path}")
        index_manifest: dict[str, Any] = {}
    else:
        index_manifest = _read_json(index_manifest_path)
        retrieval_manifest_sha = sha256_file(retrieval_manifest_path)
        index_ok = True
        if index_manifest.get("schema_version") != INDEX_MANIFEST_SCHEMA:
            index_ok = False
        if index_manifest.get("retrieval_profile") != expected_profile:
            index_ok = False
        if index_manifest.get("corpus_manifest_path") != "retrieval_manifest.json":
            index_ok = False
        if index_manifest.get("corpus_manifest_sha256") != retrieval_manifest_sha:
            index_ok = False
        if index_manifest.get("corpus_sha256") != manifest.get("corpus_sha256"):
            index_ok = False
        if int(index_manifest.get("indexed_document_count", -1)) != len(rows):
            index_ok = False
        if index_manifest.get("source_corpus_role") != manifest.get("corpus_role"):
            index_ok = False
        if not index_manifest.get("embedding_asset_sha256"):
            index_ok = False
        sparse_sha = _sha256_tree(corpus_root / "index" / "sparse")
        dense_sha = _sha256_tree(corpus_root / "index" / "dense")
        if sparse_sha != index_manifest.get("sparse_index_sha256"):
            index_ok = False
        if dense_sha != index_manifest.get("dense_index_sha256"):
            index_ok = False
        checks["retrieval_index_hash_verified"] = index_ok
        if not index_ok:
            _append_error(errors, "retrieval_index_hash_mismatch", "Index manifest, hash, or document count differs.")

    checks["retrieval_index_is_harmeme_train_only"] = all(
        [
            checks["fhm_absent_from_retrieval_databases"],
            checks["memotion_absent_from_retrieval_databases"],
            checks["source_validation_absent_from_retrieval_databases"],
            unknown_count == 0,
            checks["retrieval_split_hash_verified"],
            checks["retrieval_corpus_hash_verified"],
            checks["retrieval_index_hash_verified"],
        ]
    )
    return _root_audit_result(
        corpus_root,
        checks,
        errors,
        manifest=manifest,
        index_manifest=index_manifest,
        observed={
            "document_count": len(rows),
            "query_origin_sample_count": len(query_keys),
            "forbidden_origin_counts": {
                "harmeme_validation": validation_count,
                "facebook": fhm_count,
                "memotion": memotion_count,
                "unknown": unknown_count,
            },
        },
    )


def resolve_retrieval_runtime(
    config: dict[str, Any],
    *,
    disabled: bool = False,
    require_paper_eligible: bool = True,
) -> dict[str, Any]:
    """Resolve and verify the active runtime profile without legacy fallback."""

    if disabled:
        return {
            "enabled": False,
            "retrieval_profile": None,
            "corpus_paths": [],
            "read_only": True,
            "paper_eligible": True,
            "reason": "retrieval_disabled_by_experiment_contract",
        }
    retrieval_cfg = config.get("retrieval", {}) or {}
    active = str(retrieval_cfg.get("active_profile") or "")
    if not active:
        # Compatibility for non-paper unit configs. Canonical repository config
        # always declares a profile and therefore takes the strict branch.
        return {
            "enabled": True,
            "retrieval_profile": "unverified_legacy_config",
            "corpus_paths": list(config.get("paths", {}).get("retrieval_corpus_paths", []) or []),
            "read_only": False,
            "paper_eligible": False,
            "verified": False,
        }
    profile_cfg = retrieval_profile_config(config, active)
    root = Path(str(profile_cfg.get("corpus_root", "")))
    source_manifest = profile_cfg.get("source_split_manifest") or DEFAULT_SOURCE_MANIFEST
    audit = audit_retrieval_root(
        root,
        expected_profile=active,
        expected_source_manifest=source_manifest,
        strict=require_paper_eligible,
    )
    if not audit["passed"]:
        codes = ", ".join(error["code"] for error in audit["errors"])
        raise RetrievalProtocolError(
            "retrieval_manifest_invalid",
            f"Active retrieval profile {active!r} failed runtime verification: {codes}",
        )
    manifest = audit["manifest"]
    index_manifest = audit["index_manifest"]
    if require_paper_eligible and (active != CANONICAL_PROFILE or not manifest.get("paper_eligible")):
        raise RetrievalProtocolError(
            "retrieval_legacy_profile_selected", "Paper runtime requires harmeme_train_v1."
        )
    corpus_manifest_path = root / "retrieval_manifest.json"
    index_manifest_path = root / "index_manifest.json"
    corpus_path = root / str(manifest["corpus_file"])
    return {
        "enabled": True,
        "verified": True,
        "retrieval_profile": active,
        "corpus_paths": [str(corpus_path)],
        "corpus_manifest_path": str(corpus_manifest_path),
        "corpus_manifest_sha256": sha256_file(corpus_manifest_path),
        "corpus_sha256": manifest["corpus_sha256"],
        "index_manifest_path": str(index_manifest_path),
        "index_manifest_sha256": sha256_file(index_manifest_path),
        "sparse_index_sha256": index_manifest["sparse_index_sha256"],
        "dense_index_sha256": index_manifest["dense_index_sha256"],
        "corpus_role": manifest["corpus_role"],
        "paper_eligible": bool(manifest["paper_eligible"]),
        "read_only": bool(profile_cfg.get("read_only", True)),
        "root": str(root),
    }


def retrieval_run_manifest_fields(
    runtime: dict[str, Any], query_cache_path: str | Path | None = None
) -> dict[str, Any]:
    """Return stable retrieval provenance fields for experiment manifests."""

    return {
        "retrieval_profile": runtime.get("retrieval_profile"),
        "retrieval_corpus_manifest_path": runtime.get("corpus_manifest_path"),
        "retrieval_corpus_manifest_sha256": runtime.get("corpus_manifest_sha256"),
        "retrieval_corpus_sha256": runtime.get("corpus_sha256"),
        "retrieval_index_manifest_path": runtime.get("index_manifest_path"),
        "retrieval_index_manifest_sha256": runtime.get("index_manifest_sha256"),
        "retrieval_sparse_index_sha256": runtime.get("sparse_index_sha256"),
        "retrieval_dense_index_sha256": runtime.get("dense_index_sha256"),
        "retrieval_corpus_role": runtime.get("corpus_role"),
        "retrieval_paper_eligible": runtime.get("paper_eligible"),
        "retrieval_read_only": runtime.get("read_only", True),
        "retrieval_query_cache_path": str(query_cache_path) if query_cache_path else None,
    }


def retrieval_profile_config(config: dict[str, Any], profile: str) -> dict[str, Any]:
    profiles = (config.get("retrieval", {}) or {}).get("profiles", {}) or {}
    value = profiles.get(profile)
    if not isinstance(value, dict):
        raise RetrievalProtocolError("retrieval_profile_missing", f"Unknown retrieval profile: {profile}")
    return value


def retrieval_status(
    *,
    profile: str = CANONICAL_PROFILE,
    registry_path: str | Path = "configs/experiment_registry.yaml",
    config_path: str | Path = "configs/config.yaml",
) -> dict[str, Any]:
    """Return a compact configured-profile status without mutating artifacts."""

    return audit_retrieval_profile(
        registry_path=registry_path,
        config_path=config_path,
        profile=profile,
        strict=False,
    )


def _select_train_queries(
    allowed_rows: dict[str, dict[str, Any]], replay_sources: list[ReplaySource]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, set[str]]]]:
    query_rows: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    origins: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {
            "sample_ids": set(),
            "datasets": set(),
            "domains": set(),
            "query_ids": set(),
            "query_types": set(),
        }
    )
    for source in replay_sources:
        if source.dataset_name not in ALLOWED_DATASETS:
            raise RetrievalProtocolError(
                "blocked_retrieval_corpus_rebuild",
                f"Replay source is not an allowed HarMeme dataset: {source.dataset_name}",
            )
        if not source.query_cache.is_file():
            raise RetrievalProtocolError(
                "blocked_retrieval_corpus_rebuild", f"Query cache is missing: {source.query_cache}"
            )
        for raw in _iter_jsonl(source.query_cache):
            sample_id = str(raw.get("id", ""))
            sample_key = f"{source.dataset_name}::{sample_id}"
            if sample_key not in allowed_rows:
                continue
            if sample_key in seen_samples:
                raise RetrievalProtocolError(
                    "blocked_retrieval_corpus_rebuild", f"Duplicate query row for {sample_key}"
                )
            query = str(raw.get("query", "")).strip()
            if not query:
                raise RetrievalProtocolError(
                    "blocked_retrieval_corpus_rebuild", f"Empty cached query for {sample_key}"
                )
            query_id = _query_id(source.dataset_name, sample_id, query)
            document_ids: list[str] = []
            for candidate in raw.get("topk", []) or []:
                document_id = str(candidate.get("kid", "")).strip()
                if not document_id:
                    continue
                document_ids.append(document_id)
                origin = origins[document_id]
                origin["sample_ids"].add(sample_key)
                origin["datasets"].add(source.dataset_name)
                origin["domains"].add(source.domain)
                origin["query_ids"].add(query_id)
                origin["query_types"].add(QUERY_TYPE)
            query_rows.append(
                {
                    "query_id": query_id,
                    "query_type": QUERY_TYPE,
                    "sample_id": sample_id,
                    "sample_key": sample_key,
                    "original_dataset": source.dataset_name,
                    "domain": source.domain,
                    "source_partition": "train",
                    "query": query,
                    "retrieved_document_ids": sorted(set(document_ids)),
                    "source_cache_path": str(source.query_cache),
                    "contains_labels": False,
                }
            )
            seen_samples.add(sample_key)
    query_rows.sort(key=lambda row: row["sample_key"])
    return query_rows, origins


def _load_source_documents(path: Path, selected_ids: set[str]) -> dict[str, dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    for raw in _iter_jsonl(path):
        document_id = str(raw.get("kid", ""))
        if document_id not in selected_ids:
            continue
        if document_id in documents:
            raise RetrievalProtocolError(
                "blocked_retrieval_corpus_rebuild", f"Duplicate source document ID: {document_id}"
            )
        documents[document_id] = {
            "document_id": document_id,
            "kid": document_id,
            "text": str(raw.get("text", "")),
            "title": str(raw.get("title", "")),
            "source": str(raw.get("source", "wikipedia")),
            "url": str(raw.get("url", "")),
            "pageid": raw.get("pageid"),
            "revid": raw.get("revid"),
            "rev_timestamp": raw.get("rev_timestamp"),
        }
    return documents


def _corpus_row(
    document_id: str, document: dict[str, Any], origin: dict[str, set[str]]
) -> dict[str, Any]:
    source_identifier = document.get("url") or document_id
    return {
        "document_id": document_id,
        "text": document.get("text", ""),
        "title": document.get("title", ""),
        "source_name": document.get("source", "wikipedia"),
        "source_uri_or_identifier": source_identifier,
        "retrieval_source_type": "wikipedia_sentence",
        "origin_sample_ids": sorted(origin["sample_ids"]),
        "origin_original_datasets": sorted(origin["datasets"]),
        "origin_domains": sorted(origin["domains"]),
        "origin_split": "train",
        "origin_query_ids": sorted(origin["query_ids"]),
        "origin_query_types": sorted(origin["query_types"]),
        "is_dataset_internal_meme": False,
        "is_external_document": True,
        "created_from_harmeme_train_only": True,
        "page_id": document.get("pageid"),
        "revision_id": document.get("revid"),
        "revision_timestamp": document.get("rev_timestamp"),
    }


def _build_indexes(root: Path, rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    sparse_dir = root / "index" / "sparse"
    dense_dir = root / "index" / "dense"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    dense_dir.mkdir(parents=True, exist_ok=True)
    sparse_path = sparse_dir / "document_terms.jsonl"
    sparse_rows = []
    for row in rows:
        terms = _terms(str(row.get("text", "")))
        sparse_rows.append(
            {
                "document_id": row["document_id"],
                "length": len(terms),
                "term_frequencies": dict(sorted(Counter(terms).items())),
            }
        )
    _write_jsonl(sparse_path, sparse_rows)
    _write_json(
        sparse_dir / "metadata.json",
        {
            "backend": "bm25_token_statistics_v1",
            "document_count": len(rows),
            "tokenizer": "lowercase_alphanumeric_apostrophe_v1",
        },
    )

    ids_path = dense_dir / "document_ids.jsonl"
    vectors_path = dense_dir / "embeddings.f32"
    _write_jsonl(ids_path, [{"row": index, "document_id": row["document_id"]} for index, row in enumerate(rows)])
    with vectors_path.open("wb") as handle:
        for row in rows:
            vector = hashed_vector(str(row.get("text", "")), dim=DENSE_DIM).detach().cpu().float().tolist()
            handle.write(struct.pack(f"<{DENSE_DIM}f", *vector))
    _write_json(
        dense_dir / "metadata.json",
        {
            "backend": "hashed_vector_binary_v1",
            "document_count": len(rows),
            "dimension": DENSE_DIM,
            "dtype": "float32_little_endian",
        },
    )
    return (
        ["index/sparse/document_terms.jsonl", "index/sparse/metadata.json"],
        ["index/dense/document_ids.jsonl", "index/dense/embeddings.f32", "index/dense/metadata.json"],
    )


def _replay_sources(profile_cfg: dict[str, Any]) -> list[ReplaySource]:
    values = profile_cfg.get("replay_sources", []) or []
    sources = [
        ReplaySource(
            dataset_name=str(item.get("dataset_name", "")),
            domain=str(item.get("domain", "")),
            query_cache=Path(str(item.get("query_cache", ""))),
        )
        for item in values
        if isinstance(item, dict)
    ]
    if {source.dataset_name for source in sources} != ALLOWED_DATASETS:
        raise RetrievalProtocolError(
            "blocked_retrieval_corpus_rebuild",
            "Canonical profile must declare exactly the harm_c and harm_p replay caches.",
        )
    return sources


def _validate_source_rows(rows: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for row in rows:
        key = str(row.get("sample_key", ""))
        dataset = str(row.get("original_dataset", ""))
        if not key or not row.get("sample_id") or dataset not in ALLOWED_DATASETS:
            raise RetrievalProtocolError(
                "retrieval_unknown_origin", f"Invalid immutable train row: {row}"
            )
        if key != f"{dataset}::{row['sample_id']}":
            raise RetrievalProtocolError(
                "retrieval_unknown_origin", f"Sample key does not match dataset/sample ID: {key}"
            )
        if key in seen:
            raise RetrievalProtocolError("retrieval_unknown_origin", f"Duplicate train key: {key}")
        seen.add(key)


def _atomic_install(temporary: Path, destination: Path, *, force: bool) -> None:
    if not destination.exists():
        os.replace(temporary, destination)
        return
    if not force:
        raise RetrievalProtocolError("retrieval_output_exists", f"Output already exists: {destination}")
    backup = destination.with_name(f".{destination.name}.backup")
    if backup.exists():
        raise RetrievalProtocolError(
            "retrieval_atomic_install_blocked", f"Stale backup prevents atomic replacement: {backup}"
        )
    os.replace(destination, backup)
    try:
        os.replace(temporary, destination)
    except Exception:
        os.replace(backup, destination)
        raise
    shutil.rmtree(backup)


def _write_checksums(root: Path) -> None:
    paths = [path for path in root.rglob("*") if path.is_file() and path.name != "checksums.sha256"]
    lines = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in sorted(paths)]
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_summary(root: Path, audit: dict[str, Any]) -> dict[str, Any]:
    manifest = audit.get("manifest", {}) or {}
    index = audit.get("index_manifest", {}) or {}
    return {
        "status": audit.get("status"),
        "output_root": str(root),
        "retrieval_profile": manifest.get("retrieval_profile"),
        "source_split_manifest_sha256": manifest.get("source_split_manifest_sha256"),
        "query_origin_sample_count": manifest.get("query_origin_sample_count"),
        "query_origin_sample_coverage": manifest.get("query_origin_sample_coverage"),
        "document_count": manifest.get("document_count"),
        "corpus_sha256": manifest.get("corpus_sha256"),
        "sparse_index_sha256": index.get("sparse_index_sha256"),
        "dense_index_sha256": index.get("dense_index_sha256"),
        "paper_eligible": manifest.get("paper_eligible"),
    }


def _root_audit_result(
    root: Path,
    checks: dict[str, bool],
    errors: list[dict[str, Any]],
    *,
    manifest: dict[str, Any] | None = None,
    index_manifest: dict[str, Any] | None = None,
    observed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    passed = not errors
    return {
        "schema_version": "retrieval_audit_v1",
        "retrieval_root": str(root),
        "passed": passed,
        "status": "pass" if passed else "fail",
        "checks": checks,
        "errors": errors,
        "manifest": manifest or {},
        "index_manifest": index_manifest or {},
        "observed": observed or {},
    }


def _audit_failure(code: str, message: str) -> dict[str, Any]:
    return {
        "schema_version": "retrieval_audit_v1",
        "passed": False,
        "status": "fail",
        "checks": {},
        "errors": [{"code": code, "message": message}],
    }


def _append_error(errors: list[dict[str, Any]], code: str, message: str) -> None:
    if not any(item.get("code") == code and item.get("message") == message for item in errors):
        errors.append({"code": code, "message": message})


def _query_id(dataset_name: str, sample_id: str, query: str) -> str:
    digest = hashlib.sha256(f"{dataset_name}\0{sample_id}\0{query}".encode("utf-8")).hexdigest()
    return f"query:{digest[:24]}"


def _terms(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _sha256_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_tree(root: Path) -> str:
    if not root.is_dir():
        return ""
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
        digest.update(b"\0")
    return digest.hexdigest()


def _reproducible_timestamp() -> str:
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _same_path_text(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return str(left) == str(right)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


__all__ = [
    "CANONICAL_PROFILE",
    "CANONICAL_ROLE",
    "INDEX_MANIFEST_SCHEMA",
    "LEGACY_PROFILE",
    "MANIFEST_SCHEMA",
    "RetrievalProtocolError",
    "audit_retrieval_profile",
    "audit_retrieval_root",
    "build_retrieval_corpus",
    "resolve_retrieval_runtime",
    "retrieval_profile_config",
    "retrieval_run_manifest_fields",
    "retrieval_status",
]
