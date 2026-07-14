"""Typed schemas shared by research orchestration, aggregation, and exports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


ALLOWED_STATUSES = {
    "not_implemented",
    "ready",
    "running",
    "completed",
    "failed",
    "blocked_dependency",
    "blocked_checkpoint",
    "blocked_license_review",
    "blocked_human_annotation",
    "blocked_api_credentials",
    "skipped_budget",
    "unsupported_current_protocol",
}


@dataclass
class ExperimentSpec:
    """Normalized registry entry used by the orchestration layer."""

    id: str
    family: str
    group: str
    display_name: str
    model: str
    adapter: str
    priority: str
    enabled: bool
    status: str
    source_train_datasets: list[str]
    source_validation_datasets: list[str]
    heldout_test_datasets: list[str]
    tasks: list[str]
    seeds: list[int]
    split_manifest: str
    dependencies: list[str]
    execution_type: str
    environment_name: str
    estimated_cost: str
    estimated_gpu_memory: str | int | float | None
    official_paper_url: str | None
    official_code_url: str | None
    official_code_status: str
    implementation_policy: str
    paper_targets: list[str]
    notes: list[str] = field(default_factory=list)
    ablation: str | None = None
    knowledge_mode: str | None = None
    catalog_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchResultRow:
    """Canonical long-form result row with explicit missing values."""

    experiment_id: str
    suite: str
    family: str
    group: str
    model: str
    condition: str
    dataset: str
    dataset_family: str
    domain: str
    domain_role: str
    annotation_provenance: str
    seed: int | None
    task: str
    metric: str
    value: float | None
    valid_n: int | None
    total_n: int | None
    coverage: float | None
    unknown_count: int | None
    ambiguous_count: int | None
    masked_count: int | None
    class_distribution: dict[str, int] | None
    status: str
    split_sha256: str | None
    config_sha256: str | None
    code_sha256: str | None
    adapter_version: str | None
    external_commit: str | None
    asset_sha256: str | None
    training_strategy: str | None
    runtime_seconds: float | None
    gpu_hours: float | None
    peak_gpu_memory_mb: float | None
    timestamp: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


REQUIRED_RUN_ARTIFACTS = (
    "run_manifest.json",
    "resolved_config.yaml",
    "environment.json",
    "training_log.json",
    "runtime.json",
    "validation_predictions.jsonl",
    "test_predictions.jsonl",
    "metrics.json",
    "thresholds.json",
    "complexity.json",
    "pipeline_audit_report.json",
    "pipeline_audit_report.md",
)


__all__ = ["ALLOWED_STATUSES", "ExperimentSpec", "REQUIRED_RUN_ARTIFACTS", "ResearchResultRow"]
