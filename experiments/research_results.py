"""Canonical aggregation for locked-protocol research runs."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from experiments.registry import load_experiment_registry
from experiments.research_schemas import ResearchResultRow
from utils.io import write_json, write_jsonl


METRIC_ALIASES = {
    "accuracy": ("harmfulness", "accuracy"),
    "macro_f1": ("harmfulness", "macro_f1"),
    "weighted_f1": ("harmfulness", "weighted_f1"),
    "roc_auc": ("harmfulness", "roc_auc"),
    "harmfulness_accuracy": ("harmfulness", "accuracy"),
    "harmfulness_macro_f1": ("harmfulness", "macro_f1"),
    "harmfulness_weighted_f1": ("harmfulness", "weighted_f1"),
    "harmfulness_roc_auc": ("harmfulness", "roc_auc"),
    "target_presence_macro_f1": ("target_presence", "macro_f1"),
    "target_granularity_macro_f1": ("target_granularity", "macro_f1"),
    "protected_attribute_micro_f1": ("protected_attribute", "micro_f1"),
    "protected_attribute_macro_f1": ("protected_attribute", "macro_f1"),
    "intent_primary_macro_f1": ("intent_primary", "macro_f1"),
    "tactic_multimodal_relation_macro_f1": ("tactic_multimodal_relation", "macro_f1"),
    "tactic_rhetorical_macro_f1_logits_only": ("tactic_rhetorical", "macro_f1_logits_only"),
    "tactic_rhetorical_micro_f1_logits_only": ("tactic_rhetorical", "micro_f1_logits_only"),
    "tactic_rhetorical_none_f1": ("tactic_rhetorical", "none_f1"),
    "tactic_rhetorical_exact_match_ratio": ("tactic_rhetorical", "exact_match_ratio"),
    "evidence_hit_at_k": ("evidence", "hit_at_k"),
    "evidence_precision_at_k": ("evidence", "precision_at_k"),
}


def aggregate_research_results(
    *,
    output_root: str = "result",
    registry_path: str = "configs/experiment_registry.yaml",
) -> dict[str, Any]:
    """Scan canonical runs and write long, summary, status, and coverage tables."""

    root = Path(output_root)
    registry = load_experiment_registry(registry_path)
    experiments = registry.get("experiments", {}) or {}
    long_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    run_dirs = sorted((root / "research_runs").glob("*/*/seed_*"))
    for run_dir in run_dirs:
        manifest = _read_json(run_dir / "run_manifest.json")
        metrics = _read_json(run_dir / "metrics.json")
        suite = run_dir.parts[-3]
        experiment_id = str(manifest.get("experiment_id") or run_dir.parent.name)
        spec = experiments.get(experiment_id, {}) or {}
        seed = _int_or_none(manifest.get("seed") or run_dir.name.removeprefix("seed_"))
        audit = _read_json(run_dir / "pipeline_audit_report.json")
        completion_status = str(manifest.get("completion_status") or "partial")
        status_rows.append(
            {
                "suite": suite,
                "experiment_id": experiment_id,
                "seed": seed,
                "status": completion_status,
                "audit_passed": audit.get("passed"),
                "run_dir": str(run_dir),
            }
        )
        seen: set[tuple[str, str]] = set()
        for source_key, (task, metric_name) in METRIC_ALIASES.items():
            if source_key not in metrics or (task, metric_name) in seen:
                continue
            value = _float_or_none(metrics.get(source_key))
            seen.add((task, metric_name))
            row = ResearchResultRow(
                experiment_id=experiment_id,
                suite=suite,
                family=str(spec.get("family", manifest.get("family", "unknown"))),
                group=str(spec.get("group", manifest.get("group", "unknown"))),
                model=str(spec.get("display_name", experiment_id)),
                condition=str(spec.get("ablation") or spec.get("knowledge_mode") or "full"),
                dataset="facebook",
                dataset_family="fhm",
                domain="facebook",
                domain_role="heldout_target_test",
                annotation_provenance="original_fhm_label" if task == "harmfulness" else "agent_silver_structured_evaluation",
                seed=seed,
                task=task,
                metric=metric_name,
                value=value,
                valid_n=_metric_count(metrics, task, "valid_n"),
                total_n=_metric_count(metrics, task, "total_n"),
                coverage=_float_or_none(metrics.get(f"{task}_coverage")),
                unknown_count=_metric_count(metrics, task, "unknown_count"),
                ambiguous_count=_metric_count(metrics, task, "ambiguous_count"),
                masked_count=_metric_count(metrics, task, "masked_count"),
                class_distribution=metrics.get(f"{task}_class_distribution") if isinstance(metrics.get(f"{task}_class_distribution"), dict) else None,
                status="completed" if completion_status == "complete" and audit.get("passed") else "unaudited_or_failed",
                split_sha256=manifest.get("source_split_manifest_sha256") or manifest.get("split_sha256"),
                config_sha256=manifest.get("config_sha256"),
                code_sha256=manifest.get("code_sha256"),
                adapter_version=manifest.get("adapter_version"),
                external_commit=manifest.get("external_commit"),
                asset_sha256=_asset_hash(manifest),
                training_strategy=manifest.get("training_strategy") or spec.get("execution_type"),
                runtime_seconds=_float_or_none(_read_json(run_dir / "runtime.json").get("wall_seconds")),
                gpu_hours=None,
                peak_gpu_memory_mb=None,
                timestamp=manifest.get("created_at_utc"),
            ).to_dict()
            long_rows.append(row)
        for label, value in (metrics.get("tactic_rhetorical_per_label_f1_logits_only", {}) or {}).items():
            if not isinstance(value, (int, float)):
                continue
            base = next((row for row in reversed(long_rows) if row["experiment_id"] == experiment_id and row["task"] == "tactic_rhetorical"), None)
            if base:
                per_label = dict(base)
                per_label["metric"] = f"per_label_f1::{label}"
                per_label["value"] = float(value)
                long_rows.append(per_label)
    for experiment_id, spec in experiments.items():
        if any(row["experiment_id"] == experiment_id for row in status_rows):
            continue
        status_rows.append(
            {
                "suite": None,
                "experiment_id": experiment_id,
                "seed": None,
                "status": spec.get("status"),
                "audit_passed": None,
                "run_dir": None,
            }
        )

    results_root = root / "aggregates"
    results_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(results_root / "all_results.jsonl", long_rows)
    _write_csv(results_root / "all_results.csv", long_rows)
    (results_root / "all_results.json").write_text(json.dumps(long_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    seed_rows = summarize_seeds(long_rows)
    _write_csv(results_root / "results_mean_std.csv", seed_rows)
    coverage_rows = coverage_table(long_rows)
    _write_csv(results_root / "coverage_results.csv", coverage_rows)
    _write_csv(results_root / "experiment_status.csv", status_rows)
    significance = paired_significance_eligibility(long_rows)
    _write_csv(results_root / "significance_results.csv", significance)
    _write_csv(results_root / "main_results.csv", [row for row in long_rows if row.get("family") == "E1" and row.get("task") == "harmfulness"])
    _write_csv(results_root / "structured_results.csv", [row for row in long_rows if row.get("task") != "harmfulness"])
    _write_csv(results_root / "ablation_results.csv", [row for row in long_rows if row.get("group") == "core_ablation"])
    _write_csv(results_root / "knowledge_results.csv", [row for row in long_rows if row.get("group") == "knowledge_comparison"])
    _write_csv(
        results_root / "runtime_results.csv",
        [
            {key: row.get(key) for key in ("experiment_id", "suite", "seed", "runtime_seconds", "gpu_hours", "peak_gpu_memory_mb", "status")}
            for row in _unique_runs(long_rows)
        ],
    )
    _write_literature_metadata(results_root / "literature_reported_results.csv", experiments)
    summary = {
        "run_count": len(run_dirs),
        "metric_row_count": len(long_rows),
        "mean_std_row_count": len(seed_rows),
        "status_counts": _counts(status_rows, "status"),
        "paths": {
            "long": str(results_root / "all_results.csv"),
            "mean_std": str(results_root / "results_mean_std.csv"),
            "coverage": str(results_root / "coverage_results.csv"),
            "status": str(results_root / "experiment_status.csv"),
            "significance_eligibility": str(results_root / "significance_results.csv"),
        },
    }
    write_json(results_root / "aggregation_summary.json", summary)
    return summary


def summarize_seeds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate only completed numeric rows sharing the same comparison key."""

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = ("experiment_id", "family", "group", "model", "condition", "dataset", "task", "metric")
    for row in rows:
        if row.get("status") == "completed" and _float_or_none(row.get("value")) is not None:
            grouped[tuple(row.get(key) for key in keys)].append(row)
    output = []
    for group_key, group in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        values = [float(row["value"]) for row in group]
        output.append(
            {
                **dict(zip(keys, group_key)),
                "seed_count": len(values),
                "seeds": " ".join(str(row.get("seed")) for row in sorted(group, key=lambda item: int(item.get("seed") or 0))),
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else None,
                "valid_n_mean": _mean_optional([row.get("valid_n") for row in group]),
                "total_n_mean": _mean_optional([row.get("total_n") for row in group]),
                "coverage_mean": _mean_optional([row.get("coverage") for row in group]),
            }
        )
    return output


def coverage_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: row.get(key) for key in ("experiment_id", "seed", "dataset", "task", "valid_n", "total_n", "coverage", "unknown_count", "ambiguous_count", "masked_count", "class_distribution", "annotation_provenance", "status")}
        for row in rows
        if row.get("task") != "harmfulness"
    ]


def paired_significance_eligibility(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Report whether seed-paired tests are valid; do not manufacture p-values."""

    by_metric: dict[tuple[str, str, str], dict[str, dict[int, str | None]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row.get("status") != "completed" or row.get("seed") is None or row.get("value") is None:
            continue
        key = (str(row.get("dataset")), str(row.get("task")), str(row.get("metric")))
        by_metric[key][str(row.get("experiment_id"))][int(row["seed"])] = row.get("split_sha256")
    output = []
    for (dataset, task, metric), models in sorted(by_metric.items()):
        if "ours_full" not in models:
            continue
        for model, seeds in sorted(models.items()):
            if model == "ours_full":
                continue
            paired = sorted(
                seed
                for seed in set(models["ours_full"]) & set(seeds)
                if models["ours_full"][seed] and models["ours_full"][seed] == seeds[seed]
            )
            output.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "metric": metric,
                    "model_a": "ours_full",
                    "model_b": model,
                    "paired_seed_count": len(paired),
                    "paired_seeds": " ".join(map(str, paired)),
                    "eligible": len(paired) >= 2,
                    "reason": "ready" if len(paired) >= 2 else "requires at least two matched completed seeds with identical split hashes",
                }
            )
    return output


def _metric_count(metrics: dict[str, Any], task: str, suffix: str) -> int | None:
    value = metrics.get(f"{task}_{suffix}")
    return _int_or_none(value)


def _asset_hash(manifest: dict[str, Any]) -> str | None:
    state = manifest.get("pretrained_asset_provenance", {}) or {}
    hashes = []
    for key in ("vision", "text"):
        value = state.get(key, {}) if isinstance(state, dict) else {}
        digest = value.get("sha256") or value.get("checkpoint_sha256")
        if digest:
            hashes.append(str(digest))
    return "+".join(hashes) if hashes else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not columns:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _mean_optional(values: list[Any]) -> float | None:
    numbers = [_float_or_none(value) for value in values]
    present = [value for value in numbers if value is not None]
    return mean(present) if present else None


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    output: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "unknown"))
        output[value] = output.get(value, 0) + 1
    return dict(sorted(output.items()))


def _unique_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (row.get("suite"), row.get("experiment_id"), row.get("seed"))
        output.setdefault(key, row)
    return list(output.values())


def _write_literature_metadata(path: Path, experiments: dict[str, Any]) -> None:
    rows = [
        {
            "experiment_id": experiment_id,
            "display_name": spec.get("display_name"),
            "official_paper_url": spec.get("official_paper_url"),
            "reported_dataset": None,
            "reported_metric": None,
            "reported_value": None,
            "status": "metadata_only_no_score_imported",
        }
        for experiment_id, spec in experiments.items()
        if spec.get("adapter") == "blocked_external"
    ]
    _write_csv(path, rows)


__all__ = ["aggregate_research_results", "coverage_table", "paired_significance_eligibility", "summarize_seeds"]
