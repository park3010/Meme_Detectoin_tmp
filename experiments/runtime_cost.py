"""Runtime, memory, parameter, and fallback-rate profiling."""

from __future__ import annotations

import csv
import resource
import time
from pathlib import Path
from typing import Any

import torch

from dataset import MemeDataset
from experiments.progress import ProgressConfig, progress_iter
from module.runner import HarmfulMemePipeline
from utils.io import load_yaml, write_json


STAGES = ["stage_a", "stage_b", "stage_c", "stage_d", "stage_e"]


def run_runtime_cost_analysis(
    dataset: str = "harm_c",
    limit: int = 20,
    device: str = "cpu",
    warmup: int = 1,
    config_path: str = "configs/config.yaml",
    output_root: str = "result",
    disable_tqdm: bool = False,
    progress: ProgressConfig | None = None,
    print_components: bool = False,
) -> dict[str, Any]:
    """Profile stage latency and lightweight resource statistics."""

    cfg = load_yaml(config_path)
    cfg.setdefault("runtime", {})["device"] = device if device == "cpu" or torch.cuda.is_available() else "cpu"
    meme_dataset = MemeDataset(
        dataset_root=cfg.get("paths", {}).get("dataset_root", "dataset/source"),
        annotation_root=cfg.get("paths", {}).get("annotation_root", "dataset/annotation"),
        dataset_names=[dataset],
        keep_missing_images=True,
        limit=limit,
    )
    samples = [dict(sample) for sample in meme_dataset]
    pipeline = HarmfulMemePipeline(cfg).eval()
    if print_components:
        from experiments.components import print_pipeline_components

        print_pipeline_components(pipeline)
    progress_config = progress or ProgressConfig(disable=True if disable_tqdm else None)
    for sample in progress_iter(
        samples[:warmup],
        desc="runtime warmup",
        config=progress_config,
        position=2,
        leave=progress_config.leave_batch,
    ):
        _profile_sample(pipeline, sample)
    details = [
        _profile_sample(pipeline, sample)
        for sample in progress_iter(
            samples,
            desc="runtime measured",
            config=progress_config,
            position=2,
            leave=progress_config.leave_batch,
        )
    ]
    summary = summarize_runtime(details, pipeline, cfg)
    summary["dataset"] = dataset
    summary["sample_count"] = len(samples)
    _write_summary_row(Path(output_root) / "metrics" / "runtime_cost.csv", summary)
    write_json(Path(output_root) / "analysis" / "runtime_cost_details.json", {"summary": summary, "samples": details})
    return summary


@torch.no_grad()
def _profile_sample(pipeline: HarmfulMemePipeline, sample: dict[str, Any]) -> dict[str, Any]:
    timings: dict[str, float] = {}
    start_total = time.perf_counter()
    start = time.perf_counter()
    stage_a = pipeline.stage_a(sample)
    timings["stage_a_latency_ms"] = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    stage_b = pipeline.stage_b(stage_a, sample.get("ocr_text_full", ""))
    timings["stage_b_latency_ms"] = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    stage_c = pipeline.stage_c(stage_a, stage_b)
    timings["stage_c_latency_ms"] = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    stage_d = pipeline.stage_d(stage_a, stage_c)
    timings["stage_d_latency_ms"] = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    _ = pipeline.stage_e(stage_a, stage_c, stage_d)
    timings["stage_e_latency_ms"] = (time.perf_counter() - start) * 1000
    timings["total_latency_ms"] = (time.perf_counter() - start_total) * 1000
    return {
        "sample_id": sample.get("sample_id"),
        **timings,
        "fallback_image_encoder": "fallback" in str(stage_a.metadata.image_backend).lower(),
        "fallback_text_encoder": "fallback" in str(stage_a.metadata.text_backend).lower(),
        "fallback_retrieval": bool(stage_b.metadata.fallback_candidates_used),
        "fallback_detector": stage_a.metadata.roi_count == 0,
        "fallback_generator": any("template" in str(text).lower() for text in stage_b.generated_hypotheses),
    }


def summarize_runtime(details: list[dict[str, Any]], pipeline: HarmfulMemePipeline, cfg: dict[str, Any]) -> dict[str, Any]:
    """Aggregate per-sample runtime details."""

    summary: dict[str, Any] = {}
    for stage in STAGES:
        key = f"{stage}_latency_ms"
        values = [float(row[key]) for row in details if key in row]
        summary[key] = sum(values) / len(values) if values else None
    total = [float(row["total_latency_ms"]) for row in details if "total_latency_ms" in row]
    summary["total_latency_ms"] = sum(total) / len(total) if total else None
    trainable, frozen = _parameter_counts(pipeline)
    summary["trainable_parameter_count"] = trainable
    summary["frozen_parameter_count"] = frozen
    summary["memory_usage_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    summary["retrieval_corpus_size_bytes"] = _path_list_size(cfg.get("paths", {}).get("retrieval_corpus_paths", []) or [])
    summary["cache_size_bytes"] = _cache_size(cfg)
    for key in ["fallback_image_encoder", "fallback_text_encoder", "fallback_retrieval", "fallback_detector", "fallback_generator"]:
        summary[f"{key}_rate"] = sum(1 for row in details if row.get(key)) / max(1, len(details))
    return summary


def _parameter_counts(pipeline: HarmfulMemePipeline) -> tuple[int, int]:
    trainable = frozen = 0
    for param in pipeline.parameters():
        if param.requires_grad:
            trainable += param.numel()
        else:
            frozen += param.numel()
    return trainable, frozen


def _path_list_size(paths: list[str]) -> int:
    total = 0
    for raw in paths:
        path = Path(raw)
        if path.exists() and path.is_file():
            total += path.stat().st_size
    return total


def _cache_size(cfg: dict[str, Any]) -> int:
    root = Path(cfg.get("paths", {}).get("dataset_root", "dataset/source")) / "wiki_common" / "cache"
    if not root.exists():
        return 0
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _write_summary_row(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        if not exists:
            writer.writeheader()
        writer.writerow(summary)
