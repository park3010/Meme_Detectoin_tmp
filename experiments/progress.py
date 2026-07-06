"""Canonical progress-bar helpers used by experiment runners.

The rest of the codebase should depend on this module rather than importing
``tqdm`` directly.  That keeps CLI flags, non-interactive behavior, and fallback
handling consistent across training, suites, evaluation, and diagnostics.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class ProgressConfig:
    """Runtime policy for tqdm progress bars."""

    disable: bool | None = None
    mininterval: float = 0.5
    leave_suite: bool = True
    leave_epoch: bool = False
    leave_batch: bool = False
    dynamic_ncols: bool = True


class _PlainProgress(Iterable[T]):
    """Small tqdm-like fallback used when tqdm is disabled or unavailable."""

    def __init__(self, iterable: Iterable[T]) -> None:
        self._iterable = iterable

    def __iter__(self):
        return iter(self._iterable)

    def __enter__(self):
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def update(self, _n: int = 1) -> None:
        return None

    def close(self) -> None:
        return None

    def set_postfix(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _load_tqdm() -> Any | None:
    try:
        from tqdm.auto import tqdm

        return tqdm
    except Exception:
        return None


def make_progress(
    iterable: Iterable[T],
    *,
    desc: str = "",
    config: ProgressConfig | None = None,
    disable: bool | None = None,
    leave: bool | None = None,
    total: int | None = None,
    position: int | None = None,
    mininterval: float | None = None,
    dynamic_ncols: bool | None = None,
    **kwargs: Any,
) -> Iterable[T]:
    """Wrap an iterable with tqdm when available and requested.

    ``disable=None`` is intentionally passed through to tqdm so tqdm can
    auto-disable itself on non-TTY output.  When disabled explicitly, return the
    original iterable to preserve old behavior and keep tests simple.
    """

    progress_config = config or ProgressConfig()
    resolved_disable = progress_config.disable if disable is None else disable
    if resolved_disable is True:
        return iterable
    tqdm = _load_tqdm()
    if tqdm is None:
        return _PlainProgress(iterable)
    return tqdm(
        iterable,
        desc=desc,
        disable=resolved_disable,
        leave=progress_config.leave_batch if leave is None else leave,
        total=total,
        position=position,
        mininterval=progress_config.mininterval if mininterval is None else mininterval,
        dynamic_ncols=progress_config.dynamic_ncols if dynamic_ncols is None else dynamic_ncols,
        **kwargs,
    )


def progress_iter(
    iterable: Iterable[T],
    *,
    desc: str = "",
    config: ProgressConfig | None = None,
    disable: bool | None = None,
    leave: bool | None = None,
    total: int | None = None,
    position: int | None = None,
    mininterval: float | None = None,
    dynamic_ncols: bool | None = None,
    **kwargs: Any,
) -> Iterable[T]:
    """Backward-compatible progress wrapper."""

    return make_progress(
        iterable,
        desc=desc,
        config=config,
        disable=disable,
        leave=leave,
        total=total,
        position=position,
        mininterval=mininterval,
        dynamic_ncols=dynamic_ncols,
        **kwargs,
    )


def progress_write(message: str, *, config: ProgressConfig | None = None) -> None:
    """Write a user-facing message without corrupting active progress bars."""

    progress_config = config or ProgressConfig()
    if progress_config.disable is True:
        print(message)
        return
    tqdm = _load_tqdm()
    if tqdm is not None:
        try:
            tqdm.write(message)
            return
        except Exception:
            pass
    print(message)


def set_progress_postfix(progress: Any, **values: Any) -> None:
    """Set a tqdm postfix when the object supports it."""

    setter = getattr(progress, "set_postfix", None)
    if callable(setter):
        try:
            setter(**values)
        except Exception:
            return


def progress_config_from_flags(
    *,
    disable_tqdm: bool = False,
    tqdm_mininterval: float = 0.5,
    tqdm_leave: bool = False,
) -> ProgressConfig:
    """Build a canonical progress config from common CLI flags."""

    return ProgressConfig(
        disable=True if disable_tqdm else None,
        mininterval=float(tqdm_mininterval),
        leave_suite=True,
        leave_epoch=bool(tqdm_leave),
        leave_batch=bool(tqdm_leave),
    )


__all__ = [
    "ProgressConfig",
    "make_progress",
    "progress_config_from_flags",
    "progress_iter",
    "progress_write",
    "set_progress_postfix",
]
