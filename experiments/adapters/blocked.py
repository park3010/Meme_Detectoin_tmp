"""Non-executing adapter for unavailable or unapproved external methods."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import ExperimentAdapter


class BlockedExternalAdapter(ExperimentAdapter):
    """Expose lifecycle methods while refusing to fabricate external runs."""

    def _blocked(self) -> dict[str, Any]:
        return {
            "experiment_id": self.spec.id,
            "status": self.spec.status,
            "reason": list(self.spec.notes),
            "dependencies": list(self.spec.dependencies),
            "implementation_policy": self.spec.implementation_policy,
        }

    def prepare_data(self) -> dict[str, Any]:
        return self._blocked()

    def train_or_fit(self) -> dict[str, Any]:
        raise RuntimeError(f"Experiment {self.spec.id} is blocked: {self.spec.status}")

    def predict_validation(self) -> Path | None:
        return None

    def freeze_selection(self) -> dict[str, Any]:
        return self._blocked()

    def predict_test(self) -> Path | None:
        return None

    def evaluate(self) -> dict[str, Any]:
        return self._blocked()

    def audit(self) -> dict[str, Any]:
        return {"passed": False, **self._blocked()}

    def export(self) -> dict[str, Any]:
        return self._blocked()
