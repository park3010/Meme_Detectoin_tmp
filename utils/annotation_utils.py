"""Helpers for normalizing silver annotation payloads.

The annotation JSONL files were produced by an LLM pipeline, so these helpers
intentionally accept small schema variations instead of assuming every field is
present and perfectly typed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any


MISSING_VALUES = {None, "", "null", "none", "n/a", "na", "unknown"}


def sample_to_dict(sample: Any) -> dict[str, Any]:
    """Convert a dataset sample dataclass or mapping to a plain dictionary."""

    if isinstance(sample, Mapping):
        return dict(sample)
    if is_dataclass(sample):
        return asdict(sample)
    if hasattr(sample, "to_dict"):
        return dict(sample.to_dict())
    raise TypeError(f"Unsupported sample type for annotation normalization: {type(sample)!r}")


def unwrap_annotation(sample_or_annotation: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the annotation payload regardless of one or two wrapper layers."""

    if not sample_or_annotation:
        return {}
    obj: Any = sample_or_annotation
    if isinstance(obj, Mapping) and isinstance(obj.get("annotation"), Mapping):
        obj = obj["annotation"]
    if isinstance(obj, Mapping) and isinstance(obj.get("annotation"), Mapping):
        obj = obj["annotation"]
    return dict(obj) if isinstance(obj, Mapping) else {}


def get_block(annotation: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Return a nested annotation block or an empty dictionary."""

    value = annotation.get(name)
    return dict(value) if isinstance(value, Mapping) else {}


def normalize_key(value: Any) -> str:
    """Normalize a free-form label key for alias lookup."""

    if value is None:
        return ""
    return str(value).strip().lower().replace("-", "_").replace("/", "_").replace(" ", "_")


def is_missing(value: Any) -> bool:
    """Return True when a value should be treated as absent/unknown."""

    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in MISSING_VALUES
    if isinstance(value, (list, tuple, set)):
        return len(value) == 0
    return False


def parse_bool(value: Any, default: bool = False) -> bool:
    """Parse flexible boolean values from LLM annotation fields."""

    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "sure"}:
        return True
    if lowered in {"0", "false", "no", "n", "none", ""}:
        return False
    return default


def as_list(value: Any) -> list[Any]:
    """Coerce scalar, comma-separated, or list-like values into a list."""

    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    if isinstance(value, str):
        if not value.strip():
            return []
        separators = [",", ";", "|"]
        values = [value]
        for sep in separators:
            if sep in value:
                values = value.split(sep)
                break
        return [item.strip() for item in values if item.strip()]
    return [value]


def normalize_label(
    value: Any,
    field: str,
    config: Mapping[str, Any],
    *,
    default: str = "unknown",
) -> tuple[str, bool]:
    """Normalize a scalar label and report whether it was unknown/unmapped."""

    maps = _label_maps(config).get(field, {})
    allowed = set(str(item) for item in maps.get("allowed", []))
    aliases = {normalize_key(k): str(v) for k, v in dict(maps.get("aliases", {})).items()}

    key = normalize_key(value)

    # First, preserve explicit aliases such as "none" -> "none".
    if key in aliases:
        normalized = aliases[key]
        if allowed and normalized not in allowed:
            return "other" if "other" in allowed else default, True
        return normalized, normalized == "unknown"

    # Next, preserve valid labels even if they look like missing values.
    if allowed and key in allowed:
        return key, key == "unknown"

    # Only now treat truly missing values as unknown/default.
    if is_missing(value):
        normalized = aliases.get("", default)
        return normalized, normalized in {"unknown", default}

    normalized = key
    if allowed and normalized not in allowed:
        return "other" if "other" in allowed else default, True
    return normalized, normalized == "unknown"


def normalize_label_list(
    value: Any,
    field: str,
    config: Mapping[str, Any],
    *,
    default: str = "unknown",
) -> tuple[list[str], list[Any]]:
    """Normalize a multi-label field and return original unknown values."""

    normalized: list[str] = []
    unknown_values: list[Any] = []
    for item in as_list(value):
        label, unknown = normalize_label(item, field, config, default=default)
        if unknown and not is_missing(item):
            unknown_values.append(item)
        if label not in normalized:
            normalized.append(label)
    if not normalized:
        normalized = ["unknown"] if default == "unknown" else []
    return normalized, unknown_values


def confidence_score(confidence: str, config: Mapping[str, Any]) -> float:
    """Map normalized confidence labels to numeric scores."""

    maps = _label_maps(config).get("confidence", {})
    numeric = maps.get("numeric", {})
    try:
        return float(numeric.get(confidence, 0.0))
    except (TypeError, ValueError):
        return 0.0


def map_raw_harmfulness(raw_label: Any, dataset_name: str, config: Mapping[str, Any]) -> str:
    """Map dataset-specific raw labels to harmful/non-harmful/unknown."""

    mappings = config.get("raw_label_mapping", {}) if isinstance(config, Mapping) else {}
    dataset_map = mappings.get(dataset_name, {}) or mappings.get("default", {})
    for key in (raw_label, str(raw_label), normalize_key(raw_label)):
        if key in dataset_map:
            return str(dataset_map[key])
    return "unknown"


def raw_annotation_keys(annotation: Mapping[str, Any]) -> list[str]:
    """Return stable top-level annotation keys for provenance metadata."""

    return sorted(str(key) for key in annotation.keys())


def _label_maps(config: Mapping[str, Any]) -> Mapping[str, Any]:
    if "label_maps" in config and isinstance(config["label_maps"], Mapping):
        return config["label_maps"]
    return config
