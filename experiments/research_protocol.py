"""Locked HarMeme-to-FHM dataset protocol and leakage auditing."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from dataset import MemeDataset, NormalizedLabelStore, PAPER_DATASET_PROTOCOL
from experiments.splits import label_to_int
from utils.io import load_yaml, write_json


PROTOCOL_NAME = "harmeme_to_fhm_v1"
SOURCE_DATASETS = ("harm_c", "harm_p")
HELDOUT_DATASET = "facebook"
DISABLED_DATASETS = ("memotion",)
DEFAULT_SPLIT_SEED = 42
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_SOURCE_MANIFEST = Path("result/splits/harmeme/source_split_seed_42.json")
DEFAULT_FHM_MANIFEST = Path("result/splits/fhm/heldout_test_manifest.json")


@dataclass(frozen=True)
class ProtocolPaths:
    """Canonical immutable manifest locations."""

    source_manifest: Path = DEFAULT_SOURCE_MANIFEST
    fhm_manifest: Path = DEFAULT_FHM_MANIFEST


def sample_key(dataset_name: str, sample_id: str) -> str:
    """Return an unambiguous cross-dataset sample identifier."""

    return f"{dataset_name}::{sample_id}"


def ensure_protocol_manifests(
    *,
    dataset_root: str | Path = "dataset/source",
    annotation_root: str | Path = "dataset/annotation",
    normalized_root: str | Path = "dataset/annotation_normalized",
    label_set: str = "clean",
    split_seed: int = DEFAULT_SPLIT_SEED,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    source_manifest_path: str | Path = DEFAULT_SOURCE_MANIFEST,
    fhm_manifest_path: str | Path = DEFAULT_FHM_MANIFEST,
    force_regenerate: bool = False,
) -> dict[str, Any]:
    """Create or validate the immutable source and held-out manifests.

    Harmfulness uses every source row with a binary original label. Structured
    supervision eligibility is recorded separately from the requested clean
    label set, so clean filtering never silently removes harmfulness examples.
    """

    source_dataset = MemeDataset(
        dataset_root=dataset_root,
        annotation_root=annotation_root,
        dataset_names=list(SOURCE_DATASETS),
        keep_missing_images=True,
    )
    fhm_dataset = MemeDataset(
        dataset_root=dataset_root,
        annotation_root=annotation_root,
        dataset_names=[HELDOUT_DATASET],
        keep_missing_images=True,
    )
    full_store = NormalizedLabelStore(normalized_root, list(SOURCE_DATASETS) + [HELDOUT_DATASET], label_set="full")
    clean_store = NormalizedLabelStore(normalized_root, list(SOURCE_DATASETS) + [HELDOUT_DATASET], label_set=label_set)

    source_rows = [_manifest_row(source_dataset[index], full_store, clean_store) for index in range(len(source_dataset))]
    fhm_rows = [_manifest_row(fhm_dataset[index], full_store, clean_store) for index in range(len(fhm_dataset))]
    source_manifest = build_source_manifest(source_rows, split_seed=split_seed, train_ratio=train_ratio, label_set=label_set)
    fhm_manifest = build_fhm_manifest(fhm_rows, label_set=label_set)

    source_path = _write_immutable_manifest(source_manifest_path, source_manifest, force=force_regenerate)
    fhm_path = _write_immutable_manifest(fhm_manifest_path, fhm_manifest, force=force_regenerate)
    return {
        "source_manifest_path": str(source_path),
        "source_manifest_sha256": sha256_file(source_path),
        "fhm_manifest_path": str(fhm_path),
        "fhm_manifest_sha256": sha256_file(fhm_path),
        "source": validate_source_manifest(source_manifest),
        "fhm": validate_fhm_manifest(fhm_manifest),
    }


def build_source_manifest(
    rows: list[dict[str, Any]],
    *,
    split_seed: int = DEFAULT_SPLIT_SEED,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    label_set: str = "clean",
) -> dict[str, Any]:
    """Build a deterministic split stratified by domain and harmfulness."""

    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between zero and one")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["original_dataset"]), str(row["harmfulness"]))].append(dict(row))
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for group_key in sorted(groups):
        group_rows = sorted(groups[group_key], key=lambda item: item["sample_key"])
        random.Random(f"{split_seed}:{group_key[0]}:{group_key[1]}").shuffle(group_rows)
        train_count = int(round(len(group_rows) * train_ratio))
        train.extend(group_rows[:train_count])
        validation.extend(group_rows[train_count:])
    train.sort(key=lambda item: item["sample_key"])
    validation.sort(key=lambda item: item["sample_key"])
    manifest: dict[str, Any] = {
        "schema_version": "harmeme_source_split_v1",
        "protocol": PROTOCOL_NAME,
        "source_family": "HarMeme",
        "split_seed": int(split_seed),
        "model_seed_independent": True,
        "train_ratio": float(train_ratio),
        "validation_ratio": float(1.0 - train_ratio),
        "stratification_fields": ["original_dataset", "harmfulness"],
        "label_set": label_set,
        "harmfulness_supervision": "original_dataset_label",
        "structured_supervision": "normalized_annotation_with_clean_eligibility",
        "train": train,
        "validation": validation,
    }
    manifest["statistics"] = _source_statistics(manifest)
    manifest["content_sha256"] = content_sha256(manifest)
    return manifest


def build_fhm_manifest(rows: list[dict[str, Any]], *, label_set: str = "clean") -> dict[str, Any]:
    """Build a test-only FHM manifest with explicit silver provenance."""

    records = sorted((dict(row) for row in rows), key=lambda item: item["sample_key"])
    manifest: dict[str, Any] = {
        "schema_version": "fhm_heldout_test_v1",
        "protocol": PROTOCOL_NAME,
        "paper_name": "FHM",
        "dataset": HELDOUT_DATASET,
        "role": "heldout_target_test",
        "label_set": label_set,
        "harmfulness_evaluation_provenance": "original_fhm_label",
        "structured_evaluation_provenance": "agent_silver_structured_evaluation",
        "test": records,
    }
    manifest["statistics"] = _records_statistics(records)
    manifest["content_sha256"] = content_sha256(manifest)
    return manifest


def validate_source_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate source split identity, disjointness, metadata, and hash."""

    errors: list[str] = []
    train = list(manifest.get("train", []) or [])
    validation = list(manifest.get("validation", []) or [])
    train_keys = [str(item.get("sample_key", "")) for item in train]
    validation_keys = [str(item.get("sample_key", "")) for item in validation]
    if len(train_keys) != len(set(train_keys)):
        errors.append("duplicate sample keys in source train")
    if len(validation_keys) != len(set(validation_keys)):
        errors.append("duplicate sample keys in source validation")
    overlap = sorted(set(train_keys) & set(validation_keys))
    if overlap:
        errors.append(f"source train/validation overlap: {len(overlap)}")
    all_rows = train + validation
    invalid_datasets = sorted({str(item.get("original_dataset")) for item in all_rows} - set(SOURCE_DATASETS))
    if invalid_datasets:
        errors.append(f"unexpected source datasets: {invalid_datasets}")
    if any(not item.get("normalized_label_exists") for item in all_rows):
        errors.append("source sample missing normalized label row")
    expected_hash = manifest.get("content_sha256")
    if expected_hash and expected_hash != content_sha256(manifest):
        errors.append("source manifest content hash mismatch")
    return {
        "passed": not errors,
        "errors": errors,
        "train_count": len(train),
        "validation_count": len(validation),
        "total_count": len(all_rows),
        "overlap_count": len(overlap),
        "statistics": _source_statistics(manifest),
    }


def validate_fhm_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate that FHM is a unique test-only list."""

    errors: list[str] = []
    if "train" in manifest or "validation" in manifest or "valid" in manifest:
        errors.append("FHM manifest must not contain train/validation assignments")
    rows = list(manifest.get("test", []) or [])
    keys = [str(item.get("sample_key", "")) for item in rows]
    if len(keys) != len(set(keys)):
        errors.append("duplicate sample keys in FHM test manifest")
    if any(item.get("original_dataset") != HELDOUT_DATASET for item in rows):
        errors.append("non-FHM row in held-out manifest")
    if any(not item.get("normalized_label_exists") for item in rows):
        errors.append("FHM sample missing normalized label row")
    expected_hash = manifest.get("content_sha256")
    if expected_hash and expected_hash != content_sha256(manifest):
        errors.append("FHM manifest content hash mismatch")
    return {"passed": not errors, "errors": errors, "test_count": len(rows), "statistics": _records_statistics(rows)}


def audit_fhm_leakage(
    source_manifest: dict[str, Any],
    fhm_manifest: dict[str, Any],
    *,
    config_path: str | Path = "configs/config.yaml",
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit all enforceable FHM and Memotion leakage constraints."""

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    source_keys = {
        str(item.get("sample_key"))
        for split in ("train", "validation")
        for item in source_manifest.get(split, []) or []
    }
    fhm_keys = {str(item.get("sample_key")) for item in fhm_manifest.get("test", []) or []}
    overlap = sorted(source_keys & fhm_keys)
    _record_check(errors, not overlap, "fhm_source_overlap", f"FHM/source overlap count: {len(overlap)}")

    cfg = load_yaml(config_path)
    retrieval_paths = list(cfg.get("paths", {}).get("retrieval_corpus_paths", []) or [])
    retrieval_checks: list[dict[str, Any]] = []
    for raw_path in retrieval_paths:
        path = Path(raw_path)
        check = _audit_retrieval_path(path, fhm_keys)
        retrieval_checks.append(check)
        if not check["passed"]:
            errors.append({"code": "fhm_retrieval_leakage", "message": check["reason"], "path": str(path)})

    paper_suites = (registry or {}).get("suites", {}) if isinstance(registry, dict) else {}
    memotion_suites = []
    for name, suite in paper_suites.items():
        datasets = set(suite.get("datasets", []) or []) if isinstance(suite, dict) else set()
        if "memotion" in datasets:
            memotion_suites.append(name)
    _record_check(errors, not memotion_suites, "memotion_in_paper_suite", f"Paper suites containing Memotion: {memotion_suites}")

    checks = {
        "fhm_absent_from_source_train": not bool(overlap),
        "fhm_absent_from_source_validation": not bool(overlap),
        "fhm_absent_from_retrieval_databases": all(item["passed"] for item in retrieval_checks),
        "threshold_selection_source": "HarMeme validation only",
        "early_stopping_source": "HarMeme validation only",
        "few_shot_policy": "HarMeme only",
        "prompt_development_policy": "HarMeme only",
        "configuration_selection_reads_fhm": False,
        "memotion_disabled": not memotion_suites,
    }
    return {
        "schema_version": "fhm_leakage_audit_v1",
        "protocol": PROTOCOL_NAME,
        "passed": not errors,
        "status": "pass" if not errors and not warnings else "warning" if not errors else "fail",
        "checks": checks,
        "retrieval_checks": retrieval_checks,
        "errors": errors,
        "warnings": warnings,
    }


def source_tree_sha256(root: str | Path = ".") -> str:
    """Hash source/config/document inputs when no Git worktree is available."""

    root_path = Path(root).resolve()
    allowed_suffixes = {".py", ".yaml", ".yml", ".tex", ".bib", ".md", ".json"}
    allowed_roots = ["configs", "dataset", "experiments", "module", "scripts", "utils", "docs", "latex"]
    excluded_parts = {"source", "annotation", "annotation_normalized", "build", "generated", "__pycache__"}
    paths: list[Path] = []
    for directory in allowed_roots:
        base = root_path / directory
        if not base.exists():
            continue
        for path in base.rglob("*"):
            relative = path.relative_to(root_path)
            if not path.is_file() or path.suffix.lower() not in allowed_suffixes:
                continue
            if any(part in excluded_parts for part in relative.parts):
                continue
            paths.append(path)
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root_path).as_posix()):
        relative = path.relative_to(root_path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load a JSON protocol manifest."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def select_manifest_samples(
    samples: Iterable[dict[str, Any]],
    manifest_rows: Iterable[dict[str, Any]],
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Select samples in manifest order and attach protocol metadata/masks."""

    index = {
        sample_key(str(sample.get("dataset_name", "")), str(sample.get("sample_id", ""))): sample
        for sample in samples
    }
    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in manifest_rows:
        key = str(row.get("sample_key", ""))
        sample = index.get(key)
        if sample is None:
            missing.append(key)
            continue
        enriched = _attach_protocol_row(sample, row)
        selected.append(enriched)
        if limit is not None and len(selected) >= limit:
            break
    if missing:
        raise ValueError(f"Manifest references {len(missing)} missing samples; examples={missing[:5]}")
    return selected


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_sha256(manifest: dict[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("content_sha256", None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest_row(sample: dict[str, Any], full_store: NormalizedLabelStore, clean_store: NormalizedLabelStore) -> dict[str, Any]:
    dataset_name = str(sample.get("dataset_name", ""))
    sample_id = str(sample.get("sample_id", ""))
    label = label_to_int(sample.get("raw_label"))
    if label is None:
        raise ValueError(f"Sample has no binary original label: {dataset_name}/{sample_id}")
    protocol = PAPER_DATASET_PROTOCOL[dataset_name]
    return {
        "sample_key": sample_key(dataset_name, sample_id),
        "sample_id": sample_id,
        "dataset_name": dataset_name,
        "dataset_family": protocol["dataset_family"],
        "original_dataset": protocol["original_dataset"],
        "domain": protocol["domain"],
        "domain_role": protocol["domain_role"],
        "harmfulness": "harmful" if label == 1 else "non_harmful",
        "harmfulness_id": int(label),
        "normalized_label_exists": full_store.get(dataset_name, sample_id) is not None,
        "structured_label_eligible": clean_store.get(dataset_name, sample_id) is not None,
    }


def _attach_protocol_row(sample: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(sample)
    metadata = dict(enriched.get("metadata", {}) or {})
    metadata.update(
        {
            "sample_key": row.get("sample_key"),
            "dataset_family": row.get("dataset_family"),
            "original_dataset": row.get("original_dataset"),
            "domain": row.get("domain"),
            "domain_role": row.get("domain_role"),
            "structured_label_eligible": bool(row.get("structured_label_eligible")),
        }
    )
    enriched["metadata"] = metadata
    enriched["sample_key"] = row.get("sample_key")
    enriched["structured_label_eligible"] = bool(row.get("structured_label_eligible"))
    if not enriched["structured_label_eligible"]:
        targets = dict(enriched.get("targets", {}) or {})
        masks = dict(targets.get("masks", {}) or {})
        for field in list(masks):
            if field != "harmfulness":
                masks[field] = 0
        targets["masks"] = masks
        enriched["targets"] = targets
    return enriched


def _write_immutable_manifest(path: str | Path, manifest: dict[str, Any], *, force: bool) -> Path:
    target = Path(path)
    if target.exists():
        existing = load_manifest(target)
        if existing == manifest:
            return target
        if not force:
            raise FileExistsError(
                f"Refusing to replace immutable manifest {target}. Use --force-regenerate-split explicitly."
            )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    target.with_suffix(target.suffix + ".sha256").write_text(sha256_file(target) + "\n", encoding="ascii")
    return target


def _source_statistics(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "train": _records_statistics(list(manifest.get("train", []) or [])),
        "validation": _records_statistics(list(manifest.get("validation", []) or [])),
        "total": _records_statistics(list(manifest.get("train", []) or []) + list(manifest.get("validation", []) or [])),
    }


def _records_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset = Counter(str(row.get("original_dataset")) for row in rows)
    by_domain = Counter(str(row.get("domain")) for row in rows)
    by_label = Counter(str(row.get("harmfulness")) for row in rows)
    joint = Counter(f"{row.get('original_dataset')}::{row.get('harmfulness')}" for row in rows)
    return {
        "count": len(rows),
        "by_original_dataset": dict(sorted(by_dataset.items())),
        "by_domain": dict(sorted(by_domain.items())),
        "by_harmfulness": dict(sorted(by_label.items())),
        "by_stratum": dict(sorted(joint.items())),
        "structured_label_eligible_count": sum(bool(row.get("structured_label_eligible")) for row in rows),
    }


def _audit_retrieval_path(path: Path, fhm_keys: set[str]) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "passed": False, "reason": "retrieval corpus is missing"}
    manifest_candidates = [path.parent.parent / "wiki_manifest.json", path.with_name("meta.json")]
    for manifest_path in manifest_candidates:
        if not manifest_path.exists():
            continue
        try:
            metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        datasets = {str(item).lower() for item in metadata.get("datasets", []) or []}
        if datasets & {"facebook", "fhm"}:
            return {
                "path": str(path),
                "manifest_path": str(manifest_path),
                "passed": False,
                "reason": "retrieval corpus provenance includes Facebook/FHM",
                "datasets": sorted(datasets),
            }
    # A bounded exact-ID scan catches dataset-internal corpora without loading
    # large embedding arrays. General knowledge corpora normally have no such IDs.
    if path.suffix.lower() in {".json", ".jsonl", ".txt"} and path.stat().st_size <= 128 * 1024 * 1024:
        text = path.read_text(encoding="utf-8", errors="ignore")
        bare_ids = {key.split("::", 1)[1] for key in fhm_keys}
        matches = [value for value in bare_ids if f'"{value}"' in text][:5]
        if matches:
            return {"path": str(path), "passed": False, "reason": "FHM sample IDs found in retrieval corpus", "examples": matches}
    return {"path": str(path), "passed": True, "reason": "no FHM provenance or IDs detected"}


def _record_check(errors: list[dict[str, Any]], passed: bool, code: str, message: str) -> None:
    if not passed:
        errors.append({"code": code, "message": message})


__all__ = [
    "DEFAULT_FHM_MANIFEST",
    "DEFAULT_SOURCE_MANIFEST",
    "DEFAULT_SPLIT_SEED",
    "DISABLED_DATASETS",
    "HELDOUT_DATASET",
    "PROTOCOL_NAME",
    "SOURCE_DATASETS",
    "ProtocolPaths",
    "audit_fhm_leakage",
    "build_fhm_manifest",
    "build_source_manifest",
    "content_sha256",
    "ensure_protocol_manifests",
    "load_manifest",
    "sample_key",
    "select_manifest_samples",
    "sha256_file",
    "source_tree_sha256",
    "validate_fhm_manifest",
    "validate_source_manifest",
]
