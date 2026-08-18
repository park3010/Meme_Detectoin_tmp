"""Storage preflight and crash-safe checkpoint writes."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

import torch


def resolve_mount(path: str | Path) -> tuple[str, str]:
    target = Path(path).resolve()
    target.mkdir(parents=True, exist_ok=True)
    best = ("/", "unknown")
    try:
        entries = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
        for line in entries:
            left, right = line.split(" - ", 1)
            fields, fs_fields = left.split(), right.split()
            mount = fields[4].replace("\\040", " ")
            if target == Path(mount) or Path(mount) in target.parents:
                if len(mount) >= len(best[0]):
                    best = (mount, fs_fields[0])
    except (OSError, ValueError, IndexError):
        pass
    return best


def storage_preflight(
    output_root: str | Path,
    *,
    estimated_checkpoint_bytes: int,
    checkpoint_count: int = 2,
    temporary_write_multiplier: float = 1.25,
    minimum_free_bytes: int = 0,
    raise_on_failure: bool = True,
) -> dict[str, Any]:
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(root)
    required = max(
        int(minimum_free_bytes),
        int(max(0, estimated_checkpoint_bytes) * max(1, checkpoint_count) * temporary_write_multiplier),
    )
    mount, filesystem = resolve_mount(root)
    result = {
        "filesystem": filesystem,
        "mount_point": mount,
        "free_bytes_before_run": usage.free,
        "minimum_required_bytes": required,
        "status": "pass" if usage.free >= required else "block",
    }
    if result["status"] == "block" and raise_on_failure:
        raise RuntimeError(f"storage preflight failed: {usage.free} free bytes < {required} required bytes")
    return result


def atomic_torch_save(obj: Any, path: str | Path, *, save_fn: Callable[[Any, Any], None] = torch.save) -> Path:
    """Write a checkpoint in-place atomically; never expose a partial final file."""

    final = Path(path)
    final.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{final.name}.", suffix=".tmp", dir=final.parent)
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            save_fn(obj, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if not temp.exists() or temp.stat().st_size <= 0:
            raise OSError("temporary checkpoint is empty")
        os.replace(temp, final)
        try:
            directory_fd = os.open(final.parent, os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
        return final
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        temp.unlink(missing_ok=True)
        raise


__all__ = ["atomic_torch_save", "resolve_mount", "storage_preflight"]
