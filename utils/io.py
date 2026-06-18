"""File IO helpers for configs and JSONL results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from utils.tensor_utils import tensor_to_python


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL records, skipping blank lines."""

    file_path = Path(path)
    if not file_path.exists():
        return []
    records: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {file_path}:{line_no}: {exc}") from exc
            if isinstance(obj, dict):
                records.append(obj)
    return records


def write_jsonl(path: str | Path, records: Iterable[Any], compact_tensors: bool = True) -> None:
    """Write records as JSONL, converting dataclasses and tensors."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    max_elements = 12 if compact_tensors else None
    with file_path.open("w", encoding="utf-8") as handle:
        for record in records:
            obj = tensor_to_python(record, max_elements=max_elements)
            handle.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: Any) -> None:
    """Write a single JSON object."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(tensor_to_python(obj, max_elements=12), handle, ensure_ascii=False, indent=2)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML config with a tiny fallback for simple key/value files."""

    file_path = Path(path)
    if not file_path.exists():
        return {}
    try:
        import yaml  # type: ignore

        with file_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except ImportError:
        data = _load_simple_yaml(file_path)

    include = data.pop("include", None)
    if include:
        base = load_yaml(file_path.parent / str(include))
        return deep_update(base, data)
    return data


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries."""

    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    """Parse a very small subset of YAML used by these configs."""

    result: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, result)]
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, raw_value = line.strip().partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if raw_value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _coerce_scalar(raw_value.strip())
    return result


def _coerce_scalar(value: str) -> Any:
    if value in {"null", "None"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [] if not inner else [item.strip().strip("'\"") for item in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value.strip("'\"")
