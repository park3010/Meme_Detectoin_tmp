"""Registry-driven planning, preflight, execution, resume, and audit helpers."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.adapters import RunContext, create_adapter
from experiments.pipeline_audit import audit_baseline_run_artifacts, audit_run_artifacts
from experiments.pretrained_assets import inspect_text_asset, inspect_vision_asset
from experiments.registry import experiment_specs, load_experiment_registry, registry_summary, resolve_research_suite
from experiments.research_protocol import (
    DEFAULT_FHM_MANIFEST,
    audit_fhm_leakage,
    ensure_protocol_manifests,
    load_manifest,
    sha256_file,
    source_tree_sha256,
)
from experiments.research_schemas import REQUIRED_RUN_ARTIFACTS
from utils.io import load_yaml, write_json


PLANNING_ROOT = Path("result/research_planning")
RUNS_ROOT_NAME = "research_runs"


def plan_research(
    *,
    suite: str | None = None,
    registry_path: str = "configs/experiment_registry.yaml",
    output_root: str = "result",
) -> dict[str, Any]:
    """Resolve a suite or the complete registry without running experiments."""

    registry = load_experiment_registry(registry_path)
    plan: dict[str, Any] = {
        "created_at_utc": _now(),
        "registry": registry_summary(registry),
        "code_sha256": source_tree_sha256(),
    }
    if suite:
        plan["suite"] = resolve_research_suite(registry, suite)
    planning_root = Path(output_root) / "research_planning"
    planning_root.mkdir(parents=True, exist_ok=True)
    write_json(planning_root / (f"plan_{suite}.json" if suite else "registry_plan.json"), plan)
    _write_external_feasibility(registry, planning_root)
    return plan


def run_research_preflight(
    *,
    registry_path: str = "configs/experiment_registry.yaml",
    config_path: str = "configs/config.yaml",
    output_root: str = "result",
    strict: bool = True,
    force_regenerate_split: bool = False,
) -> dict[str, Any]:
    """Create immutable manifests and audit protocol, assets, and leakage."""

    registry = load_experiment_registry(registry_path)
    protocol = registry.get("protocol", {}) or {}
    manifests = ensure_protocol_manifests(
        label_set=str(protocol.get("label_set", "clean")),
        split_seed=int(protocol.get("split_seed", 42)),
        source_manifest_path=str(protocol.get("source_split_manifest")),
        fhm_manifest_path=str(protocol.get("heldout_test_manifest", DEFAULT_FHM_MANIFEST)),
        force_regenerate=force_regenerate_split,
    )
    source_manifest = load_manifest(manifests["source_manifest_path"])
    fhm_manifest = load_manifest(manifests["fhm_manifest_path"])
    leakage = audit_fhm_leakage(source_manifest, fhm_manifest, config_path=config_path, registry=registry)
    cfg = load_yaml(config_path)
    vision = asdict(inspect_vision_asset(cfg))
    text = asdict(inspect_text_asset(cfg))
    registry_check = registry_summary(registry)
    asset_passed = bool(vision.get("usable") and text.get("usable"))
    errors = []
    if not registry_check.get("passed"):
        errors.extend(registry_check.get("errors", []))
    if strict and not asset_passed:
        errors.append("required local pretrained assets are not usable")
    if not leakage.get("passed"):
        errors.extend(item.get("message", str(item)) for item in leakage.get("errors", []))
    result = {
        "schema_version": "harmeme_fhm_research_preflight_v1",
        "created_at_utc": _now(),
        "strict": strict,
        "passed": not errors,
        "status": "pass" if not errors else "fail",
        "registry": registry_check,
        "manifests": manifests,
        "assets": {"vision": vision, "text": text, "passed": asset_passed},
        "leakage_audit": leakage,
        "code_sha256": source_tree_sha256(),
        "errors": errors,
    }
    root = Path(output_root) / "research_planning"
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "protocol_preflight.json", result)
    write_json(root / "fhm_leakage_audit.json", leakage)
    (root / "protocol_preflight.md").write_text(_preflight_markdown(result), encoding="utf-8")
    (root / "fhm_leakage_audit.md").write_text(_leakage_markdown(leakage), encoding="utf-8")
    _write_external_feasibility(registry, root)
    return result


def execute_research_suite(
    suite: str,
    *,
    experiment_ids: list[str] | None = None,
    registry_path: str = "configs/experiment_registry.yaml",
    config_path: str = "configs/config.yaml",
    output_root: str = "result",
    execute: bool = False,
    resume: bool = False,
    force: bool = False,
    epochs: int | None = None,
    limit: int | None = None,
    device: str = "cpu",
    disable_tqdm: bool = False,
) -> dict[str, Any]:
    """Plan or explicitly execute one registered suite or an explicit subset."""

    registry = load_experiment_registry(registry_path)
    resolved = resolve_research_suite(registry, suite)
    if experiment_ids:
        requested = set(experiment_ids)
        suite_ids = {item["experiment_id"] for item in resolved["runs"]}
        unknown = sorted(requested - suite_ids)
        if unknown:
            raise ValueError(
                f"Experiments are not members of suite {suite}: {', '.join(unknown)}"
            )
        resolved["runs"] = [
            item for item in resolved["runs"] if item["experiment_id"] in requested
        ]
    protocol = registry.get("protocol", {}) or {}
    suite_limit = limit if limit is not None else resolved.get("limit")
    rows: list[dict[str, Any]] = []
    preflight = None
    if execute:
        preflight = run_research_preflight(
            registry_path=registry_path,
            config_path=config_path,
            output_root=output_root,
            strict=True,
        )
        if not preflight["passed"]:
            raise RuntimeError("Strict research preflight failed; inspect result/research_planning/protocol_preflight.json")
    specs = experiment_specs(registry)
    for item in resolved["runs"]:
        spec = specs[item["experiment_id"]]
        context = RunContext(
            suite=suite,
            seed=int(item["seed"]),
            output_root=output_root,
            config_path=config_path,
            fhm_manifest=str(protocol.get("heldout_test_manifest", DEFAULT_FHM_MANIFEST)),
            epochs=epochs,
            limit=suite_limit,
            device=device,
            disable_tqdm=disable_tqdm,
            force=force,
        )
        adapter = create_adapter(spec, context)
        if spec.status != "ready" or not spec.enabled:
            rows.append({"experiment_id": spec.id, "seed": context.seed, "status": spec.status, "action": "blocked"})
            continue
        completed = canonical_run_complete(adapter.run_dir)
        if completed and (resume or not force):
            rows.append({"experiment_id": spec.id, "seed": context.seed, "status": "completed", "action": "skipped_existing"})
            continue
        if not execute:
            rows.append({"experiment_id": spec.id, "seed": context.seed, "status": "ready", "action": "planned"})
            continue
        try:
            result = adapter.run()
            rows.append({"experiment_id": spec.id, "seed": context.seed, "status": "completed", "action": "executed", **result})
        except Exception as exc:
            rows.append({"experiment_id": spec.id, "seed": context.seed, "status": "failed", "action": "executed", "error": str(exc)})
            _write_suite_manifest(suite, rows, registry, output_root, preflight)
            raise
    return _write_suite_manifest(suite, rows, registry, output_root, preflight)


def research_status(*, output_root: str = "result", suite: str | None = None) -> dict[str, Any]:
    """Summarize canonical run directories and required artifacts."""

    root = Path(output_root) / RUNS_ROOT_NAME
    pattern = f"{suite}/*/seed_*" if suite else "*/*/seed_*"
    rows = []
    for run_dir in sorted(root.glob(pattern)):
        missing = [name for name in REQUIRED_RUN_ARTIFACTS if not (run_dir / name).exists()]
        manifest = _read_json(run_dir / "run_manifest.json")
        rows.append(
            {
                "run_dir": str(run_dir),
                "experiment_id": manifest.get("experiment_id", run_dir.parent.name),
                "seed": manifest.get("seed", run_dir.name.removeprefix("seed_")),
                "completion_status": manifest.get("completion_status", "partial"),
                "complete": not missing,
                "missing_artifacts": missing,
            }
        )
    return {"run_count": len(rows), "complete_count": sum(row["complete"] for row in rows), "runs": rows}


def audit_research_runs(*, output_root: str = "result", suite: str | None = None, strict: bool = True) -> dict[str, Any]:
    """Re-audit all discovered canonical/native runs plus global leakage."""

    status = research_status(output_root=output_root, suite=suite)
    rows = []
    for row in status["runs"]:
        run_dir = Path(row["run_dir"])
        native_roots = list((run_dir / "native" / "predictions").glob("*/*/*"))
        audit_root = native_roots[0] if native_roots else run_dir
        manifest = _read_json(audit_root / "run_manifest.json")
        if manifest.get("run_kind") == "baseline":
            audit = audit_baseline_run_artifacts(audit_root, strict=strict, require_nonempty_metrics=True)
        else:
            audit = audit_run_artifacts(audit_root, strict=strict, require_nonempty_metrics=True)
        rows.append({"run_dir": str(run_dir), "passed": audit.get("passed"), "status": audit.get("status")})
    planning = Path(output_root) / "research_planning" / "fhm_leakage_audit.json"
    leakage = _read_json(planning)
    return {
        "passed": all(row["passed"] for row in rows) and bool(leakage.get("passed")),
        "run_audits": rows,
        "leakage_audit": leakage,
    }


def canonical_run_complete(run_dir: Path) -> bool:
    """Return true only when all canonical artifacts exist and audit passed."""

    if not run_dir.exists() or any(not (run_dir / name).exists() for name in REQUIRED_RUN_ARTIFACTS):
        return False
    audit = _read_json(run_dir / "pipeline_audit_report.json")
    manifest = _read_json(run_dir / "run_manifest.json")
    return bool(audit.get("passed") and manifest.get("completion_status") == "complete")


def _write_suite_manifest(
    suite: str,
    rows: list[dict[str, Any]],
    registry: dict[str, Any],
    output_root: str,
    preflight: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        "schema_version": "research_suite_manifest_v1",
        "suite": suite,
        "created_at_utc": _now(),
        "registry_version": registry.get("registry_version"),
        "code_sha256": source_tree_sha256(),
        "preflight_passed": preflight.get("passed") if preflight else None,
        "runs": rows,
        "status_counts": _status_counts(rows),
    }
    root = Path(output_root) / "research_suites" / suite
    write_json(root / "suite_manifest.json", payload)
    return payload


def _write_external_feasibility(registry: dict[str, Any], root: Path) -> None:
    catalog = (registry.get("external_catalog", {}) or {}).get("models", {}) or {}
    rows = [
        {
            "experiment_id": experiment_id,
            "model_id": payload.get("catalog_id") or experiment_id,
            "display_name": payload.get("display_name"),
            "paper_title": payload.get("display_name"),
            "paper_url": payload.get("official_paper_url"),
            "status": payload.get("status"),
            "priority": payload.get("priority"),
            "official_repository_verification": payload.get("official_code_status"),
            "official_code_url": payload.get("official_code_url"),
            "official_code_status": payload.get("official_code_status"),
            "license_status": (catalog.get(payload.get("catalog_id") or experiment_id, {}) or {}).get(
                "license_status", "unknown_not_verified"
            ),
            "exact_commit_status": "not_pinned",
            "python_requirement": "unknown_not_assessed",
            "pytorch_requirement": "unknown_not_assessed",
            "cuda_requirement": "unknown_not_assessed",
            "required_checkpoints": "unknown_not_selected",
            "required_datasets": ["HarMeme source train/validation", "FHM held-out test"],
            "preprocessing_requirements": "requires_official_repository_review",
            "estimated_gpu_memory": payload.get("estimated_gpu_memory"),
            "estimated_storage": "unknown_not_assessed",
            "expected_runtime": payload.get("estimated_cost"),
            "harmeme_to_fhm_compatibility": "unverified_adapter_mapping",
            "expected_adapter_complexity": "unknown_until_official_code_review",
            "architecture_preservation_risks": [
                "checkpoint and preprocessing are not pinned",
                "paper protocol mapping has not been validated",
            ],
            "recommended_isolated_conda_environment": payload.get("environment_name"),
            "dependencies": payload.get("dependencies"),
            "implementation_policy": payload.get("implementation_policy"),
            "blocking_reasons": [payload.get("status"), *(payload.get("dependencies") or [])],
            "next_approval_action": "approve isolated official-repository inspection and pin license, commit, checkpoint, and environment",
            "notes": payload.get("notes"),
        }
        for experiment_id, payload in (registry.get("experiments", {}) or {}).items()
        if payload.get("adapter") == "blocked_external"
    ]
    write_json(root / "external_baseline_feasibility.json", {"models": rows})


def _preflight_markdown(result: dict[str, Any]) -> str:
    lines = ["# Research protocol preflight", "", f"Status: **{result['status']}**", ""]
    lines.extend(f"- {error}" for error in result.get("errors", []))
    lines.extend(["", "See `fhm_leakage_audit.md` for leakage-specific checks.", ""])
    return "\n".join(lines)


def _leakage_markdown(result: dict[str, Any]) -> str:
    lines = ["# FHM leakage audit", "", f"Status: **{result.get('status')}**", ""]
    for key, value in (result.get("checks", {}) or {}).items():
        lines.append(f"- `{key}`: {value}")
    if result.get("errors"):
        lines.extend(["", "## Blocking findings", ""])
        lines.extend(f"- {item.get('code')}: {item.get('message')}" for item in result["errors"])
    return "\n".join(lines) + "\n"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "audit_research_runs",
    "canonical_run_complete",
    "execute_research_suite",
    "plan_research",
    "research_status",
    "run_research_preflight",
]
