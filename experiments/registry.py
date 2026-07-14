"""Experiment registry loading, normalization, validation, and planning."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from experiments.research_schemas import ALLOWED_STATUSES, ExperimentSpec
from utils.io import load_yaml


def load_experiment_registry(
    path: str | Path = "configs/experiment_registry.yaml",
    external_catalog_path: str | Path = "configs/external_models.yaml",
) -> dict[str, Any]:
    """Load and normalize the canonical experiment registry."""

    raw = load_yaml(path)
    external_catalog = load_yaml(external_catalog_path)
    experiments: dict[str, dict[str, Any]] = {}
    for experiment_id, payload in (raw.get("experiments", {}) or {}).items():
        experiments[str(experiment_id)] = _complete_entry(str(experiment_id), dict(payload or {}), raw, None)
    catalog_models = external_catalog.get("models", {}) or {}
    for payload in raw.get("external_experiments", []) or []:
        item = dict(payload or {})
        experiment_id = str(item.get("id", ""))
        catalog_id = str(item.get("catalog_id") or experiment_id)
        item["catalog_id"] = catalog_id
        experiments[experiment_id] = _complete_entry(experiment_id, item, raw, dict(catalog_models.get(catalog_id, {}) or {}))
    normalized = dict(raw)
    normalized["experiments"] = experiments
    normalized["external_catalog"] = external_catalog
    normalized["registry_path"] = str(path)
    normalized["external_catalog_path"] = str(external_catalog_path)
    return normalized


def validate_registry(registry: dict[str, Any]) -> dict[str, Any]:
    """Validate statuses, required fields, suite references, and catalogs."""

    errors: list[str] = []
    warnings: list[str] = []
    experiments = registry.get("experiments", {}) or {}
    required = set(ExperimentSpec.__dataclass_fields__)
    ids = list(experiments)
    duplicates = [key for key, count in Counter(ids).items() if count > 1]
    if duplicates:
        errors.append(f"duplicate experiment IDs: {duplicates}")
    catalog_models = registry.get("external_catalog", {}).get("models", {}) or {}
    for experiment_id, payload in experiments.items():
        missing = sorted(required - set(payload))
        if missing:
            errors.append(f"{experiment_id}: missing fields {missing}")
        if payload.get("status") not in ALLOWED_STATUSES:
            errors.append(f"{experiment_id}: invalid status {payload.get('status')!r}")
        catalog_id = payload.get("catalog_id")
        if catalog_id and catalog_id not in catalog_models:
            errors.append(f"{experiment_id}: unknown external catalog reference {catalog_id}")
        if "memotion" in set(payload.get("source_train_datasets", []) or []):
            errors.append(f"{experiment_id}: Memotion cannot be a paper source dataset")
        if "facebook" in set(payload.get("source_train_datasets", []) or []) | set(payload.get("source_validation_datasets", []) or []):
            errors.append(f"{experiment_id}: FHM cannot be used for train/validation")
    suites = registry.get("suites", {}) or {}
    for suite_name, suite in suites.items():
        for experiment_id in suite.get("experiments", []) or []:
            if experiment_id not in experiments:
                errors.append(f"{suite_name}: unknown experiment {experiment_id}")
        if "memotion" in set(suite.get("datasets", []) or []):
            errors.append(f"{suite_name}: Memotion is disabled for paper suites")
    return {
        "passed": not errors,
        "schema_version": registry.get("schema_version"),
        "registry_version": registry.get("registry_version"),
        "experiment_count": len(experiments),
        "suite_count": len(suites),
        "status_counts": dict(sorted(Counter(str(item.get("status")) for item in experiments.values()).items())),
        "adapter_counts": dict(sorted(Counter(str(item.get("adapter")) for item in experiments.values()).items())),
        "errors": errors,
        "warnings": warnings,
    }


def experiment_specs(registry: dict[str, Any]) -> dict[str, ExperimentSpec]:
    """Convert normalized registry entries to dataclasses."""

    return {key: ExperimentSpec(**value) for key, value in (registry.get("experiments", {}) or {}).items()}


def resolve_research_suite(registry: dict[str, Any], suite_name: str) -> dict[str, Any]:
    """Resolve one suite to enabled experiment specs and seed combinations."""

    validation = validate_registry(registry)
    if not validation["passed"]:
        raise ValueError("Invalid experiment registry: " + "; ".join(validation["errors"]))
    suite = (registry.get("suites", {}) or {}).get(suite_name)
    if not isinstance(suite, dict):
        raise ValueError(f"Unknown research suite: {suite_name}")
    specs = experiment_specs(registry)
    runs = []
    suite_seeds = [int(seed) for seed in suite.get("seeds", []) or []]
    for experiment_id in suite.get("experiments", []) or []:
        spec = specs[str(experiment_id)]
        seeds = [seed for seed in suite_seeds if seed in spec.seeds] or list(spec.seeds[:1])
        for seed in seeds:
            runs.append({"experiment_id": spec.id, "seed": seed, "status": spec.status, "spec": spec.to_dict()})
    return {"name": suite_name, "description": suite.get("description", ""), "limit": suite.get("limit"), "runs": runs}


def registry_summary(registry: dict[str, Any]) -> dict[str, Any]:
    """Return a compact machine-readable plan summary."""

    validation = validate_registry(registry)
    return {
        **validation,
        "suites": {
            name: {
                "enabled": bool(suite.get("enabled", True)),
                "experiment_count": len(suite.get("experiments", []) or []),
                "seeds": list(suite.get("seeds", []) or []),
            }
            for name, suite in (registry.get("suites", {}) or {}).items()
        },
        "protocol": registry.get("protocol", {}),
    }


def _complete_entry(
    experiment_id: str,
    payload: dict[str, Any],
    registry: dict[str, Any],
    catalog: dict[str, Any] | None,
) -> dict[str, Any]:
    protocol = registry.get("protocol", {}) or {}
    catalog = catalog or {}
    status = str(payload.get("status") or catalog.get("status") or "not_implemented")
    entry = {
        "id": experiment_id,
        "family": str(payload.get("family", "E1")),
        "group": str(payload.get("group", "external")),
        "display_name": str(payload.get("display_name") or catalog.get("display_name") or experiment_id),
        "model": str(payload.get("model") or payload.get("catalog_id") or experiment_id),
        "adapter": str(payload.get("adapter", "blocked_external")),
        "priority": str(payload.get("priority") or catalog.get("priority") or "optional"),
        "enabled": bool(payload.get("enabled", False if payload.get("adapter") == "blocked_external" else True)),
        "status": status,
        "source_train_datasets": list(payload.get("source_train_datasets") or protocol.get("source_train_datasets", [])),
        "source_validation_datasets": list(payload.get("source_validation_datasets") or protocol.get("source_validation_datasets", [])),
        "heldout_test_datasets": list(payload.get("heldout_test_datasets") or protocol.get("heldout_test_datasets", [])),
        "tasks": list(payload.get("tasks") or ["harmfulness"]),
        "seeds": [int(seed) for seed in payload.get("seeds") or protocol.get("model_seeds", {}).get("one_seed", [42])],
        "split_manifest": str(payload.get("split_manifest") or protocol.get("source_split_manifest", "")),
        "dependencies": list(payload.get("dependencies") or ["explicit_approval", "isolated_environment"]),
        "execution_type": str(payload.get("execution_type", "blocked_external")),
        "environment_name": str(payload.get("environment_name") or catalog.get("environment_policy") or "isolated_environment_required"),
        "estimated_cost": str(payload.get("estimated_cost") or "unknown"),
        "estimated_gpu_memory": payload.get("estimated_gpu_memory", "unknown"),
        "official_paper_url": payload.get("official_paper_url", catalog.get("paper_url")),
        "official_code_url": payload.get("official_code_url", catalog.get("official_code_url")),
        "official_code_status": str(payload.get("official_code_status") or catalog.get("official_code_status") or "unverified"),
        "implementation_policy": str(payload.get("implementation_policy") or catalog.get("implementation_policy") or "not_implemented"),
        "paper_targets": list(payload.get("paper_targets") or ["main_baseline"]),
        "notes": [str(item) for item in payload.get("notes") or catalog.get("notes") or []],
        "ablation": payload.get("ablation"),
        "knowledge_mode": payload.get("knowledge_mode"),
        "catalog_id": payload.get("catalog_id"),
    }
    return entry


__all__ = [
    "experiment_specs",
    "load_experiment_registry",
    "registry_summary",
    "resolve_research_suite",
    "validate_registry",
]
