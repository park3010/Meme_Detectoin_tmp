"""Adapter contract shared by built-in and external research methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.research_schemas import ExperimentSpec


@dataclass
class RunContext:
    """Execution context for one registry experiment and model seed."""

    suite: str
    seed: int
    output_root: str = "result"
    config_path: str = "configs/config.yaml"
    fhm_manifest: str = "result/splits/fhm/heldout_test_manifest.json"
    epochs: int | None = None
    limit: int | None = None
    device: str = "cpu"
    disable_tqdm: bool = False
    force: bool = False

    def run_dir(self, experiment_id: str) -> Path:
        return Path(self.output_root) / "research_runs" / self.suite / experiment_id / f"seed_{self.seed}"


class ExperimentAdapter(ABC):
    """Lifecycle contract required by the research orchestration layer."""

    adapter_version = "research_adapter_v1"

    def __init__(self, spec: ExperimentSpec, context: RunContext) -> None:
        self.spec = spec
        self.context = context
        self.run_dir = context.run_dir(spec.id)

    @abstractmethod
    def prepare_data(self) -> dict[str, Any]:
        """Validate and prepare protocol inputs."""

    @abstractmethod
    def train_or_fit(self) -> dict[str, Any]:
        """Train or fit without reading held-out FHM labels for selection."""

    @abstractmethod
    def predict_validation(self) -> Path | None:
        """Return HarMeme validation predictions."""

    @abstractmethod
    def freeze_selection(self) -> dict[str, Any]:
        """Freeze thresholds and model-selection choices from validation."""

    @abstractmethod
    def predict_test(self) -> Path | None:
        """Return held-out FHM test predictions."""

    @abstractmethod
    def evaluate(self) -> dict[str, Any]:
        """Evaluate canonical outputs."""

    @abstractmethod
    def audit(self) -> dict[str, Any]:
        """Audit protocol and artifact contracts."""

    @abstractmethod
    def export(self) -> dict[str, Any]:
        """Export canonical run artifacts."""

    def run(self) -> dict[str, Any]:
        """Execute the full adapter lifecycle in protocol order."""

        self.prepare_data()
        self.train_or_fit()
        self.predict_validation()
        self.freeze_selection()
        self.predict_test()
        self.evaluate()
        audit = self.audit()
        exported = self.export()
        return {"run_dir": str(self.run_dir), "audit": audit, **exported}
