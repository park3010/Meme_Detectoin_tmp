"""Local pretrained asset inspection, manifests, and runtime verification."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from experiments.run_manifest import sha256_file
from utils.io import write_json


ASSET_SCHEMA_VERSION = "pretrained_asset_verification_v1"
VISION_MANIFEST = Path("assets/pretrained/vision/open_clip_vit_b_32/asset_manifest.json")
TEXT_MANIFEST = Path("assets/pretrained/text/deberta_v3_base/asset_manifest.json")
TEXT_REQUIRED_FILES = ["config.json", "tokenizer_config.json"]
TEXT_TOKENIZER_ALTERNATIVES = ["tokenizer.json", "spm.model", "vocab.json", "vocab.txt"]
TEXT_WEIGHT_FILES = ["model.safetensors", "pytorch_model.bin"]


@dataclass
class AssetIssue:
    """One local asset warning or blocking error."""

    code: str
    severity: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class AssetRecord:
    """Serializable inspection record for one pretrained asset."""

    asset_kind: str
    model_name: str
    source_type: str
    source_identifier: str | None
    local_path: str
    exists: bool
    usable: bool
    file_count: int
    total_size_bytes: int
    sha256: str | None
    manifest_path: str | None
    required_files: list[str]
    missing_files: list[str]
    issues: list[dict[str, Any]]


@dataclass
class AssetVerificationResult:
    """Top-level asset verification artifact."""

    schema_version: str
    passed: bool
    strict: bool
    vision: dict[str, Any]
    text: dict[str, Any]
    warnings: list[dict[str, Any]]
    errors: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_vision_asset(config: dict[str, Any]) -> AssetRecord:
    """Inspect the configured local vision checkpoint without loading a model."""

    clip_cfg = _backbone_config(config, "clip")
    path = _resolve_path(clip_cfg.get("checkpoint_path"))
    issues: list[AssetIssue] = []
    source_type = str(clip_cfg.get("asset_mode") or ("local_checkpoint" if path else "pretrained_tag"))
    required = ["checkpoint.pt"] if source_type == "local_checkpoint" else []
    exists = bool(path and path.exists())
    file_count, total_size = _file_stats(path)
    sha = sha256_file(path) if path and path.is_file() else _directory_sha256(path) if path and path.is_dir() else None
    if source_type == "local_checkpoint":
        if path is None:
            _issue(issues, "vision_checkpoint_unconfigured", "error", "Vision local_checkpoint mode requires checkpoint_path.", {})
        elif not path.exists():
            _issue(issues, "vision_checkpoint_missing", "error", "Vision checkpoint file is missing.", {"path": str(path)})
        elif not path.is_file():
            _issue(issues, "vision_checkpoint_not_file", "error", "Vision checkpoint_path must point to a file.", {"path": str(path)})
        elif path.stat().st_size <= 0:
            _issue(issues, "vision_checkpoint_empty", "error", "Vision checkpoint file is empty.", {"path": str(path)})
    elif not clip_cfg.get("allow_download", False):
        _issue(
            issues,
            "vision_nonlocal_source_disabled",
            "warning",
            "Vision asset is not local_checkpoint and downloads are disabled.",
            {"source_type": source_type},
        )
    return AssetRecord(
        asset_kind="vision",
        model_name=str(clip_cfg.get("model_name", "ViT-B-32")),
        source_type=source_type,
        source_identifier=str(clip_cfg.get("pretrained_tag")) if clip_cfg.get("pretrained_tag") else None,
        local_path=str(path) if path else "",
        exists=exists,
        usable=not any(issue.severity == "error" for issue in issues),
        file_count=file_count,
        total_size_bytes=total_size,
        sha256=sha,
        manifest_path=str(_resolve_path(clip_cfg.get("asset_manifest_path")) or VISION_MANIFEST),
        required_files=required,
        missing_files=[] if exists else required,
        issues=[asdict(issue) for issue in issues],
    )


def inspect_text_asset(config: dict[str, Any]) -> AssetRecord:
    """Inspect the configured local Hugging Face text model directory."""

    text_cfg = _backbone_config(config, "text")
    path = _resolve_path(text_cfg.get("checkpoint_path"))
    issues: list[AssetIssue] = []
    source_type = str(text_cfg.get("asset_mode") or ("local_directory" if path else "model_name"))
    exists = bool(path and path.exists())
    file_count, total_size = _file_stats(path)
    sha = _directory_sha256(path) if path and path.is_dir() else sha256_file(path) if path and path.is_file() else None
    missing = []
    if source_type == "local_directory":
        if path is None:
            _issue(issues, "text_directory_unconfigured", "error", "Text local_directory mode requires checkpoint_path.", {})
        elif not path.exists():
            _issue(issues, "text_directory_missing", "error", "Text model directory is missing.", {"path": str(path)})
        elif not path.is_dir():
            _issue(issues, "text_path_not_directory", "error", "Text checkpoint_path must point to a directory.", {"path": str(path)})
        else:
            missing.extend(name for name in TEXT_REQUIRED_FILES if not (path / name).exists())
            if not any((path / name).exists() for name in TEXT_TOKENIZER_ALTERNATIVES):
                missing.append("tokenizer.json|spm.model|vocab.json|vocab.txt")
            if not _has_weight_file(path):
                missing.append("model.safetensors|pytorch_model.bin|sharded weights")
            if missing:
                _issue(issues, "text_required_files_missing", "error", "Text model directory is incomplete.", {"path": str(path), "missing": missing})
    elif not text_cfg.get("allow_download", False):
        _issue(
            issues,
            "text_nonlocal_source_disabled",
            "warning",
            "Text asset is not local_directory and downloads are disabled.",
            {"source_type": source_type},
        )
    return AssetRecord(
        asset_kind="text",
        model_name=str(text_cfg.get("model_name", "microsoft/deberta-v3-base")),
        source_type=source_type,
        source_identifier=str(text_cfg.get("model_name")) if text_cfg.get("model_name") else None,
        local_path=str(path) if path else "",
        exists=exists,
        usable=not any(issue.severity == "error" for issue in issues),
        file_count=file_count,
        total_size_bytes=total_size,
        sha256=sha,
        manifest_path=str(_resolve_path(text_cfg.get("asset_manifest_path")) or TEXT_MANIFEST),
        required_files=[*TEXT_REQUIRED_FILES, "tokenizer asset", "model weights"],
        missing_files=missing,
        issues=[asdict(issue) for issue in issues],
    )


def verify_pretrained_assets(config: dict[str, Any], *, strict: bool) -> AssetVerificationResult:
    """Inspect local assets and instantiate runtime adapters to verify loading."""

    vision_record = inspect_vision_asset(config)
    text_record = inspect_text_asset(config)
    vision_runtime = _vision_runtime_state(config)
    text_runtime = _text_runtime_state(config)
    warnings: list[AssetIssue] = []
    errors: list[AssetIssue] = []
    _collect_record_issues(vision_record, strict, warnings, errors)
    _collect_record_issues(text_record, strict, warnings, errors)
    _collect_runtime_issues("vision", vision_runtime, strict, warnings, errors)
    _collect_runtime_issues("text", text_runtime, strict, warnings, errors)
    if strict:
        if not vision_record.usable:
            _issue(errors, "vision_asset_unusable", "error", "Vision asset inspection failed.", asdict(vision_record))
        if not text_record.usable:
            _issue(errors, "text_asset_unusable", "error", "Text asset inspection failed.", asdict(text_record))
    return AssetVerificationResult(
        schema_version=ASSET_SCHEMA_VERSION,
        passed=not errors,
        strict=bool(strict),
        vision={"asset": asdict(vision_record), "runtime": vision_runtime},
        text={"asset": asdict(text_record), "runtime": text_runtime},
        warnings=[asdict(issue) for issue in warnings],
        errors=[asdict(issue) for issue in errors],
    )


def write_asset_manifest(record: AssetRecord, *, output_path: str | Path, runtime_readiness: dict[str, Any] | None = None) -> Path:
    """Write one asset_manifest.json for the inspected asset."""

    path = Path(output_path)
    manifest = {
        "schema_version": "pretrained_asset_manifest_v1",
        "asset": asdict(record),
    }
    if record.asset_kind == "vision":
        manifest.update(
            {
                "asset_kind": "vision",
                "model_name": record.model_name,
                "asset_mode": record.source_type,
                "checkpoint_path": record.local_path,
                "checkpoint_sha256": record.sha256,
                "expected_loader": "open_clip_factory_local_path",
                "expected_pretrained_source_id": record.local_path or record.source_identifier,
                "runtime_compatibility_verified": bool((runtime_readiness or {}).get("checkpoint_compatibility_verified", False)),
                "runtime_readiness": runtime_readiness or {},
            }
        )
    write_json(path, manifest)
    return path


def build_asset_provenance(config: dict[str, Any], runtime_state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build run-manifest provenance combining asset inspection and runtime state."""

    runtime_state = runtime_state or {}
    vision_record = inspect_vision_asset(config)
    text_record = inspect_text_asset(config)
    vision_runtime = runtime_state.get("vision", {}) if isinstance(runtime_state, dict) else {}
    text_runtime = runtime_state.get("text", {}) if isinstance(runtime_state, dict) else {}
    return {
        "vision": _asset_provenance_item(config, "clip", vision_record, vision_runtime),
        "text": _asset_provenance_item(config, "text", text_record, text_runtime),
    }


def init_asset_layout(config: dict[str, Any]) -> list[Path]:
    """Create the project-relative empty pretrained asset layout."""

    paths = [
        Path("assets/pretrained/vision/open_clip_vit_b_32"),
        Path("assets/pretrained/vision/open_clip_vit_b_32/cache"),
        Path("assets/pretrained/text/deberta_v3_base"),
        Path("assets/pretrained/text/deberta_v3_base/cache"),
    ]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
        (path / ".gitkeep").touch(exist_ok=True)
    for record in [inspect_vision_asset(config), inspect_text_asset(config)]:
        manifest_path = Path(record.manifest_path) if record.manifest_path else None
        if manifest_path:
            write_asset_manifest(record, output_path=manifest_path)
    return paths


def write_verification_artifacts(result: AssetVerificationResult, *, output_root: str | Path, profile: str, write_manifests: bool = False) -> dict[str, Path]:
    """Write asset verification artifacts under result/preflight/<profile>/."""

    out = Path(output_root) / "preflight" / profile
    out.mkdir(parents=True, exist_ok=True)
    audit_path = out / "pretrained_asset_audit.json"
    write_json(audit_path, result.to_dict())
    paths = {"pretrained_asset_audit": audit_path}
    if write_manifests:
        for kind in ["vision", "text"]:
            asset = result.to_dict()[kind]["asset"]
            record = AssetRecord(**asset)
            if record.manifest_path:
                paths[f"{kind}_asset_manifest"] = write_asset_manifest(
                    record,
                    output_path=record.manifest_path,
                    runtime_readiness=result.to_dict()[kind].get("runtime", {}),
                )
    return paths


def _vision_runtime_state(config: dict[str, Any]) -> dict[str, Any]:
    clip_cfg = _backbone_config(config, "clip")
    try:
        from module.backbone.vision import CLIPWrapper

        wrapper = CLIPWrapper(
            prefer_pretrained=bool(clip_cfg.get("prefer_pretrained", False)),
            model_name=str(clip_cfg.get("model_name", "ViT-B-32")),
            device=str(config.get("runtime", {}).get("device", "cpu")),
            pretrained_tag=clip_cfg.get("pretrained_tag"),
            checkpoint_path=clip_cfg.get("checkpoint_path"),
            cache_dir=clip_cfg.get("cache_dir"),
            local_files_only=bool(clip_cfg.get("local_files_only", True)),
            allow_download=bool(clip_cfg.get("allow_download", False)),
            asset_mode=clip_cfg.get("asset_mode"),
        )
        return wrapper.readiness_state()
    except Exception as exc:
        return {"weights_loaded": False, "fallback_used": True, "random_initialization_used": False, "load_error": _short_error(exc)}


def _text_runtime_state(config: dict[str, Any]) -> dict[str, Any]:
    text_cfg = _backbone_config(config, "text")
    try:
        from module.backbone.text import TextEncoderWrapper

        wrapper = TextEncoderWrapper(
            prefer_transformers=bool(text_cfg.get("prefer_transformers", False)),
            model_name=str(text_cfg.get("model_name", "microsoft/deberta-v3-base")),
            device=str(config.get("runtime", {}).get("device", "cpu")),
            checkpoint_path=text_cfg.get("checkpoint_path"),
            cache_dir=text_cfg.get("cache_dir"),
            local_files_only=bool(text_cfg.get("local_files_only", True)),
            allow_download=bool(text_cfg.get("allow_download", False)),
            asset_mode=text_cfg.get("asset_mode"),
        )
        return wrapper.readiness_state()
    except Exception as exc:
        return {"weights_loaded": False, "fallback_used": True, "load_error": _short_error(exc)}


def _asset_provenance_item(config: dict[str, Any], key: str, record: AssetRecord, runtime: dict[str, Any]) -> dict[str, Any]:
    cfg = _backbone_config(config, key)
    return {
        "asset_mode": cfg.get("asset_mode"),
        "model_name": cfg.get("model_name"),
        "resolved_path": record.local_path,
        "sha256": record.sha256,
        "weights_loaded": bool(runtime.get("weights_loaded", False)),
        "fallback_used": bool(runtime.get("fallback_used", True)),
        "random_initialization_used": bool(runtime.get("random_initialization_used", False)),
        "weights_source": runtime.get("weights_source"),
        "checkpoint_compatibility_verified": bool(runtime.get("checkpoint_compatibility_verified", False)),
        "checkpoint_format": runtime.get("checkpoint_format"),
        "matched_parameter_ratio": runtime.get("matched_parameter_ratio"),
        "shape_mismatch_count": runtime.get("shape_mismatch_count"),
        "compatibility_failure_reason": runtime.get("compatibility_failure_reason"),
        "asset_manifest_path": record.manifest_path,
        "load_error": runtime.get("load_error"),
    }


def _collect_record_issues(record: AssetRecord, strict: bool, warnings: list[AssetIssue], errors: list[AssetIssue]) -> None:
    for issue_obj in record.issues:
        issue = AssetIssue(**issue_obj)
        target = errors if strict and issue.severity == "error" else warnings
        target.append(issue)


def _collect_runtime_issues(kind: str, state: dict[str, Any], strict: bool, warnings: list[AssetIssue], errors: list[AssetIssue]) -> None:
    target = errors if strict else warnings
    if not state.get("weights_loaded"):
        _issue(target, f"{kind}_runtime_weights_not_loaded", "error" if strict else "warning", f"{kind} runtime did not load pretrained weights.", state)
    if state.get("fallback_used"):
        _issue(target, f"{kind}_runtime_fallback_used", "error" if strict else "warning", f"{kind} runtime used fallback features.", state)
    if kind == "vision" and state.get("random_initialization_used"):
        _issue(target, "vision_runtime_random_initialization", "error" if strict else "warning", "Vision runtime used random initialization.", state)
    if kind == "vision":
        ratio = state.get("matched_parameter_ratio")
        shape_mismatches = int(state.get("shape_mismatch_count") or 0)
        reason = str(state.get("compatibility_failure_reason") or state.get("load_error") or "")
        if state.get("weights_loaded") and not state.get("checkpoint_compatibility_verified"):
            _issue(target, "vision_checkpoint_compatibility_unverified", "error" if strict else "warning", "Vision weights were not compatibility verified.", state)
        if strict and not state.get("checkpoint_compatibility_verified"):
            _issue(target, _vision_compatibility_code(reason), "error", "Vision checkpoint compatibility was not verified.", state)
        if strict and ratio is not None and float(ratio) < 0.99:
            _issue(target, "vision_checkpoint_key_mismatch", "error", "Vision checkpoint matched too little of the configured architecture.", state)
        if strict and shape_mismatches > 0:
            _issue(target, "vision_checkpoint_shape_mismatch", "error", "Vision checkpoint contains tensor shape mismatches.", state)


def _backbone_config(config: dict[str, Any], key: str) -> dict[str, Any]:
    backbone = config.get("backbone", config.get("backbones", {})) if isinstance(config, dict) else {}
    return dict(backbone.get(key, {}) or {})


def _resolve_path(value: Any) -> Path | None:
    if value is None or value == "":
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def _file_stats(path: Path | None) -> tuple[int, int]:
    if path is None or not path.exists():
        return 0, 0
    if path.is_file():
        return 1, path.stat().st_size
    files = [item for item in path.rglob("*") if item.is_file()]
    return len(files), sum(item.stat().st_size for item in files)


def _directory_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_dir():
        return None
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        rel = file_path.relative_to(path).as_posix()
        digest.update(rel.encode("utf-8"))
        file_hash = sha256_file(file_path) or ""
        digest.update(file_hash.encode("utf-8"))
    return digest.hexdigest()


def _has_weight_file(path: Path) -> bool:
    if any((path / name).exists() for name in TEXT_WEIGHT_FILES):
        return True
    return bool(list(path.glob("model-*.safetensors")) or list(path.glob("pytorch_model-*.bin")))


def _issue(issues: list[AssetIssue], code: str, severity: str, message: str, context: dict[str, Any]) -> None:
    issues.append(AssetIssue(code=code, severity=severity, message=message, context=context))


def _short_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc)[:300]}"


def _vision_compatibility_code(reason: str) -> str:
    lowered = reason.lower()
    if "shape" in lowered:
        return "vision_checkpoint_shape_mismatch"
    if "zero_matched" in lowered or "key_mismatch" in lowered or "coverage" in lowered:
        return "vision_checkpoint_key_mismatch"
    if "factory" in lowered:
        return "vision_factory_local_load_failed"
    if "missing" in lowered:
        return "vision_asset_missing"
    if "empty" in lowered:
        return "vision_asset_zero_bytes"
    return "vision_asset_exists_but_runtime_incompatible"


__all__ = [
    "AssetIssue",
    "AssetRecord",
    "AssetVerificationResult",
    "inspect_vision_asset",
    "inspect_text_asset",
    "verify_pretrained_assets",
    "write_asset_manifest",
    "build_asset_provenance",
    "init_asset_layout",
    "write_verification_artifacts",
]
