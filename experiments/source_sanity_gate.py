"""Persistent, source-only gradient gate for tiny-overfit diagnostics."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from experiments.research_protocol import sha256_file

GATE_SCHEMA = "source_sanity_gradient_gate_v1"
ACTIVE_SPLIT = Path("result/splits/harmeme/source_split_seed_42_v2.json")
ACTIVE_SPLIT_SHA256 = "1995075ba474345702ee590bc9e291522c6ebaee5f941fc1e924a867fc64e6bf"
CODE_PATHS = (
    "experiments/source_sanity.py",
    "experiments/source_sanity_gate.py",
    "experiments/source_tiny_overfit.py",
    "experiments/gradient_forensics.py",
    "experiments/train.py",
    "experiments/storage_safety.py",
    "module/runner.py",
    "module/losses.py",
    "module/knowledge_filter_verifier.py",
    "module/backbone/vision.py",
    "module/backbone/text.py",
)


def _canonical_local_path(path: str | Path) -> str:
    value = Path(path)
    absolute = value.resolve()
    try:
        return str(absolute.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(absolute)


def source_sanity_code_sha256(paths: tuple[str, ...] = CODE_PATHS) -> str:
    digest = hashlib.sha256()
    for value in sorted(paths):
        path = Path(value)
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    data = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    _atomic_bytes(path, data)
    return path


def write_gradient_gate(
    *,
    output_root: str | Path,
    config_path: str | Path,
    gradient_report_path: str | Path,
    report: dict[str, Any],
    device_requested: str,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    root = Path(output_root).resolve()
    config_path = Path(config_path)
    report_path = Path(gradient_report_path)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    first_physical = int(visible.split(",")[0]) if visible.split(",")[0].strip().isdigit() else None
    payload = {
        "schema_version": GATE_SCHEMA,
        "passed": bool(report.get("passed")),
        "decision": report.get("decision"),
        "ready_for_overfit_32": bool(report.get("passed") and report.get("decision") == "READY_FOR_OVERFIT_32"),
        "ready_for_1seed": False,
        "source_only": True,
        "device_requested": device_requested,
        "cuda_available": bool(torch.cuda.is_available()),
        "physical_gpu_id": first_physical,
        "visible_cuda_device": 0 if torch.cuda.is_available() and str(device_requested).startswith("cuda") else None,
        "output_root": str(root),
        "source_split_manifest_path": str(ACTIVE_SPLIT),
        "source_split_manifest_sha256": sha256_file(ACTIVE_SPLIT),
        "diagnostic_manifest_sha256": report.get("diagnostic_manifest_sha256"),
        "config_path": _canonical_local_path(config_path),
        "config_sha256": sha256_file(config_path),
        "code_sha256": source_sanity_code_sha256(),
        "gradient_report_path": _canonical_local_path(report_path),
        "gradient_report_sha256": sha256_file(report_path),
        "all_formal_tasks_tested": set((report.get("valid_n") or {})) == {
            "harmfulness", "target_presence", "target_granularity", "intent_primary",
            "tactic_rhetorical", "tactic_multimodal_relation",
        } and all(int(value) > 0 for value in (report.get("valid_n") or {}).values()),
        "optimizer_membership_passed": bool(report.get("optimizer_membership_passed")),
        "parameter_updates_passed": bool(report.get("parameter_updates_passed")),
        "dimensions_passed": bool(report.get("dimensions_passed")),
        "nan_or_inf": bool(report.get("nan_or_inf")),
        "fhm_or_memotion_accessed": False,
        "scientific_checkpoint_written": False,
        "created_at_utc": created_at_utc or datetime.now(timezone.utc).isoformat(),
    }
    gate = root / "gates" / "gradient_check_gate.json"
    atomic_write_json(gate, payload)
    _atomic_bytes(Path(f"{gate}.sha256"), f"{sha256_file(gate)}  {gate.name}\n".encode("ascii"))
    return payload


def validate_gradient_gate(
    *, output_root: str | Path, config_path: str | Path,
    expected_diagnostic_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    root = Path(output_root).resolve()
    gate_path = root / "gates" / "gradient_check_gate.json"
    reasons: list[str] = []
    if not gate_path.is_file():
        return {"passed": False, "reasons": ["gradient_gate_missing"], "gate_path": str(gate_path)}
    try:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
    except Exception:
        return {"passed": False, "reasons": ["gradient_gate_invalid_json"], "gate_path": str(gate_path)}
    checksum_path = Path(f"{gate_path}.sha256")
    if not checksum_path.is_file() or checksum_path.read_text().split()[0] != sha256_file(gate_path): reasons.append("gradient_gate_checksum_mismatch")
    if gate.get("schema_version") != GATE_SCHEMA: reasons.append("gradient_gate_schema_unsupported")
    if gate.get("output_root") != str(root): reasons.append("gradient_gate_output_root_mismatch")
    if gate.get("passed") is not True or gate.get("decision") != "READY_FOR_OVERFIT_32" or gate.get("ready_for_overfit_32") is not True: reasons.append("gradient_gate_not_passed")
    if gate.get("ready_for_1seed") is not False or gate.get("source_only") is not True: reasons.append("gradient_gate_scope_invalid")
    if gate.get("cuda_available") is not True or not str(gate.get("device_requested", "")).startswith("cuda"): reasons.append("gradient_gate_cuda_unverified")
    if gate.get("source_split_manifest_path") != str(ACTIVE_SPLIT) or gate.get("source_split_manifest_sha256") != ACTIVE_SPLIT_SHA256 or sha256_file(ACTIVE_SPLIT) != ACTIVE_SPLIT_SHA256: reasons.append("gradient_gate_split_hash_mismatch")
    if expected_diagnostic_manifest_sha256 and gate.get("diagnostic_manifest_sha256") != expected_diagnostic_manifest_sha256: reasons.append("gradient_gate_diagnostic_manifest_mismatch")
    report_path = Path(str(gate.get("gradient_report_path", "")))
    if not report_path.is_file() or (report_path.is_file() and sha256_file(report_path) != gate.get("gradient_report_sha256")): reasons.append("gradient_report_hash_mismatch")
    if gate.get("optimizer_membership_passed") is not True: reasons.append("gradient_gate_optimizer_failed")
    if gate.get("parameter_updates_passed") is not True: reasons.append("gradient_gate_parameter_update_failed")
    if gate.get("dimensions_passed") is not True: reasons.append("gradient_gate_dimensions_failed")
    if gate.get("nan_or_inf") is not False: reasons.append("gradient_gate_nonfinite")
    if gate.get("fhm_or_memotion_accessed") is not False: reasons.append("gradient_gate_fhm_access_detected")
    if gate.get("scientific_checkpoint_written") is not False: reasons.append("gradient_gate_scientific_checkpoint_detected")
    config_path = Path(config_path)
    if gate.get("config_path") != _canonical_local_path(config_path) or gate.get("config_sha256") != sha256_file(config_path): reasons.append("gradient_gate_config_hash_mismatch")
    if gate.get("code_sha256") != source_sanity_code_sha256(): reasons.append("gradient_gate_code_hash_mismatch")
    if gate.get("all_formal_tasks_tested") is not True: reasons.append("gradient_gate_incomplete_tasks")
    return {"passed": not reasons, "reasons": sorted(set(reasons)), "gate_path": str(gate_path), "gate_sha256": sha256_file(gate_path), "gate": gate}


__all__ = ["GATE_SCHEMA", "atomic_write_json", "source_sanity_code_sha256", "write_gradient_gate", "validate_gradient_gate"]
