"""Small progress-bar helpers used by experiment runners."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

T = TypeVar("T")


def progress_iter(
    iterable: Iterable[T],
    *,
    desc: str,
    disable: bool = False,
    leave: bool = False,
    total: int | None = None,
) -> Iterable[T]:
    """Wrap an iterable with tqdm when available and enabled."""

    if disable:
        return iterable
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, desc=desc, leave=leave, total=total)
    except Exception:
        return iterable
