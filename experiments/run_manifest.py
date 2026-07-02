"""Run manifest helpers for reproducible experiment protocol tracking."""

from __future__ import annotations

import hashlib
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.ablation_configs import (
    LOGITS_LOSSES,
    component_state_for_ablation,
    default_component_state,
    get_ablation_contract,
)
from utils.io import write_json


RUN_MANIFEST_SCHEMA = "experiment_run_manifest_v1"


def current_command() -> str:
    """Return the current Python command line in shell-readable form."""

    return " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])


def sha256_file(path: str | Path | None) -> str | None:
    """Return a SHA-256 digest for an existing file, otherwise None."""

    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_manifest(
    *,
    suite_name: str | None,
    run_kind: str,
    run_name: str,
    dataset: str,
    seed: int,
    config_path: str | Path,
    split_file: str | Path | None,
    requested_command: str | None = None,
    ablation_name: str | None = None,
    component_state: dict[str, bool] | None = None,
    expected_active_logits_losses: list[str] | None = None,
    expected_disabled_losses: list[str] | None = None,
    expected_knowledge_mode: str | None = None,
    expected_evidence_mode: str | None = None,
    completion_status: str = "complete",
    audit: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the standard JSON run manifest object."""

    contract = get_ablation_contract(ablation_name) if ablation_name else None
    if contract:
        expected_active_logits_losses = expected_active_logits_losses or list(contract.expected_active_logits_losses)
        expected_disabled_losses = expected_disabled_losses or list(contract.expected_proxy_or_disabled_losses)
        expected_knowledge_mode = expected_knowledge_mode or contract.expected_knowledge_mode
        expected_evidence_mode = expected_evidence_mode or contract.expected_evidence_mode
        component_state = component_state or component_state_for_ablation(contract.name)
    else:
        expected_active_logits_losses = expected_active_logits_losses or list(LOGITS_LOSSES)
        expected_disabled_losses = expected_disabled_losses or []
        expected_knowledge_mode = expected_knowledge_mode or "verified"
        expected_evidence_mode = expected_evidence_mode or "internal_external_evidence"
        component_state = component_state or default_component_state()

    manifest: dict[str, Any] = {
        "schema": RUN_MANIFEST_SCHEMA,
        "suite_name": suite_name,
        "run_kind": run_kind,
        "run_name": run_name,
        "dataset": dataset,
        "seed": int(seed),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "split_file": str(split_file) if split_file else None,
        "split_sha256": sha256_file(split_file),
        "requested_command": requested_command or current_command(),
        "ablation_contract": contract.to_dict() if contract else None,
        "component_state": component_state,
        "expected_active_logits_losses": expected_active_logits_losses,
        "expected_disabled_losses": expected_disabled_losses,
        "expected_knowledge_mode": expected_knowledge_mode,
        "expected_evidence_mode": expected_evidence_mode,
        "completion_status": completion_status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if audit is not None:
        manifest["audit"] = _compact_audit(audit)
    if extra:
        manifest.update(extra)
    return manifest


def write_run_manifest(output_dir: str | Path, manifest: dict[str, Any]) -> Path:
    """Write run_manifest.json under one run output directory."""

    path = Path(output_dir) / "run_manifest.json"
    write_json(path, manifest)
    return path


def update_run_manifest(output_dir: str | Path, updates: dict[str, Any]) -> dict[str, Any]:
    """Merge updates into an existing run manifest and write it back."""

    import json

    path = Path(output_dir) / "run_manifest.json"
    manifest: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            manifest = loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError:
            manifest = {}
    manifest.update(updates)
    write_json(path, manifest)
    return manifest


def _compact_audit(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": bool(audit.get("passed")),
        "status": audit.get("status"),
        "warning_count": len(audit.get("warnings", [])),
        "error_count": len(audit.get("errors", [])),
    }

