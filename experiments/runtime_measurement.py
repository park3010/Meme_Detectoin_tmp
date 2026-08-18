"""Phase-level wall/GPU instrumentation for future canonical runs."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator

import torch


class RuntimeMeasurement:
    """Measure named phases with synchronized CUDA boundaries when applicable."""

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.seconds: dict[str, float] = {}
        self.started = time.perf_counter()
        if self.cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

    @property
    def cuda(self) -> bool:
        return self.device.type == "cuda" and torch.cuda.is_available()

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if self.cuda: torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        try:
            yield
        finally:
            if self.cuda: torch.cuda.synchronize(self.device)
            self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - start

    def report(self) -> dict[str, Any]:
        total = time.perf_counter() - self.started
        allocated = torch.cuda.max_memory_allocated(self.device) if self.cuda else None
        reserved = torch.cuda.max_memory_reserved(self.device) if self.cuda else None
        return {
            "runtime_seconds_total": total,
            **{f"{name}_seconds": value for name, value in self.seconds.items()},
            "peak_gpu_memory_allocated_mb": allocated / 1024**2 if allocated is not None else None,
            "peak_gpu_memory_reserved_mb": reserved / 1024**2 if reserved is not None else None,
            "gpu_hours": total * torch.cuda.device_count() / 3600 if self.cuda else None,
            "offline_corpus_build_included": False,
        }


__all__ = ["RuntimeMeasurement"]
