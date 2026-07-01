"""Knowledge-source comparison experiments for the full framework."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.ablation_configs import KNOWLEDGE_MODES
from experiments.ablation_runner import append_metric_row, knowledge_analysis_record, run_framework_variant
from experiments.prediction_io import save_predictions_and_metrics
from utils.io import write_jsonl


def run_knowledge_comparison(
    dataset_name: str,
    mode: str,
    seed: int = 42,
    config_path: str = "configs/config.yaml",
    split_file: str | None = None,
    output_root: str = "result",
    limit: int | None = None,
    disable_tqdm: bool = False,
    print_components: bool = False,
) -> dict[str, Any]:
    """Run one knowledge-source mode and save predictions, metrics, and examples."""

    if mode not in KNOWLEDGE_MODES:
        raise ValueError(f"Unsupported knowledge mode: {mode}")
    predictions, metrics, analysis = run_framework_variant(
        dataset_name=dataset_name,
        model_name=f"knowledge_{mode}",
        seed=seed,
        config_path=config_path,
        split_file=split_file,
        output_root=output_root,
        limit=limit,
        knowledge_mode=mode,
        fusion_mode="task_aware_gate_verified",
        analysis_builder=knowledge_analysis_record,
        disable_tqdm=disable_tqdm,
        print_components=print_components,
    )
    metrics.update(_knowledge_summary(analysis))
    output_dir = Path(output_root) / "predictions" / dataset_name / f"knowledge_{mode}" / str(seed)
    save_predictions_and_metrics(output_dir, predictions, metrics)
    append_metric_row(Path(output_root) / "metrics" / "knowledge_comparison.csv", _metric_row(dataset_name, seed, mode, metrics))
    write_jsonl(Path(output_root) / "analysis" / "knowledge_comparison_examples.jsonl", analysis, compact_tensors=True)
    return metrics


def _knowledge_summary(analysis: list[dict[str, Any]]) -> dict[str, Any]:
    if not analysis:
        return {"accepted_knowledge_count": 0.0, "fallback_knowledge_ratio": 0.0}
    accepted = [float(row.get("accepted_knowledge_count", 0.0)) for row in analysis]
    fallback = [float(row.get("fallback_knowledge_ratio", 0.0)) for row in analysis]
    return {
        "accepted_knowledge_count": sum(accepted) / len(accepted),
        "fallback_knowledge_ratio": sum(fallback) / len(fallback),
    }


def _metric_row(dataset: str, seed: int, mode: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "seed": seed,
        "knowledge_mode": mode,
        "harmfulness_macro_f1": metrics.get("macro_f1") or metrics.get("harmfulness_macro_f1"),
        "target_macro_f1": metrics.get("target_granularity_macro_f1"),
        "intent_macro_f1": metrics.get("intent_primary_macro_f1"),
        "tactic_macro_f1": metrics.get("tactic_multimodal_relation_macro_f1"),
        "evidence_hit_at_k": metrics.get("evidence_hit_at_k"),
        "accepted_knowledge_count": metrics.get("accepted_knowledge_count"),
        "fallback_knowledge_ratio": metrics.get("fallback_knowledge_ratio"),
    }
