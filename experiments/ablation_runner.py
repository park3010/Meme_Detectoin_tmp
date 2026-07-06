"""Stage-wise ablation and fusion comparison runners."""

from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import torch

from dataset import MemeDataset
from experiments.ablation_configs import (
    ABLATION_MODES,
    FUSION_MODES,
    AblationConfig,
    component_state_for_ablation,
    get_ablation_config,
    get_ablation_contract,
)
from experiments.evaluation import compute_harmfulness_metrics
from experiments.prediction_io import save_predictions_and_metrics, stage_outputs_to_prediction_record
from experiments.progress import ProgressConfig, progress_iter
from experiments.run_manifest import build_run_manifest, current_command, write_run_manifest
from experiments.splits import build_splits_for_dataset, label_to_int, load_split_file, save_splits, split_samples
from experiments.evaluation import evaluate_structured_predictions
from module.runner import HarmfulMemePipeline
from module.external_knowledge_acquisition import QueryBundle, StageBMetadata, StageBOutput
from module.knowledge_filter_verifier import StageCMetadata, StageCOutput
from utils.io import load_yaml, write_jsonl
from utils.seed import set_seed


def run_ablation_experiment(
    dataset_name: str,
    ablation_name: str,
    seed: int = 42,
    config_path: str = "configs/config.yaml",
    split_file: str | None = None,
    output_root: str = "result",
    limit: int | None = None,
    disable_tqdm: bool = False,
    progress: ProgressConfig | None = None,
    print_components: bool = False,
    device: str = "cpu",
    suite_name: str | None = None,
    requested_command: str | None = None,
) -> dict[str, Any]:
    """Run one stage-wise ablation on the test split."""

    config = get_ablation_config(ablation_name)
    predictions, metrics, _ = run_framework_variant(
        dataset_name=dataset_name,
        model_name=f"ablation_{ablation_name}",
        seed=seed,
        config_path=config_path,
        split_file=split_file,
        output_root=output_root,
        limit=limit,
        ablation=config,
        disable_tqdm=disable_tqdm,
        progress=progress,
        print_components=print_components,
        device=device,
    )
    output_dir = Path(output_root) / "predictions" / dataset_name / f"ablation_{ablation_name}" / str(seed)
    save_predictions_and_metrics(output_dir, predictions, metrics)
    write_run_manifest(
        output_dir,
        build_run_manifest(
            suite_name=suite_name,
            run_kind="ablation",
            run_name=f"ablation_{config.name}",
            dataset=dataset_name,
            seed=seed,
            config_path=config_path,
            split_file=_resolved_split_path(dataset_name, seed, split_file, output_root),
            requested_command=requested_command or current_command(),
            ablation_name=config.name,
            component_state=component_state_for_ablation(config.name),
            extra={
                "ablation_name": config.name,
                "training_strategy": "evaluation_time_diagnostic",
                "diagnostic_note": "Evaluation-time diagnostic variant; not a core paper comparison.",
            },
        ),
    )
    append_metric_row(Path(output_root) / "metrics" / "ablation.csv", _ablation_metric_row(dataset_name, seed, ablation_name, metrics))
    return metrics


def run_fusion_experiment(
    dataset_name: str,
    fusion_mode: str,
    seed: int = 42,
    config_path: str = "configs/config.yaml",
    split_file: str | None = None,
    output_root: str = "result",
    limit: int | None = None,
    disable_tqdm: bool = False,
    progress: ProgressConfig | None = None,
    print_components: bool = False,
    device: str = "cpu",
    suite_name: str | None = None,
    requested_command: str | None = None,
) -> dict[str, Any]:
    """Run one fusion/gating comparison mode."""

    if fusion_mode not in FUSION_MODES:
        raise ValueError(f"Unsupported fusion mode: {fusion_mode}")
    predictions, metrics, analysis = run_framework_variant(
        dataset_name=dataset_name,
        model_name=f"fusion_{fusion_mode}",
        seed=seed,
        config_path=config_path,
        split_file=split_file,
        output_root=output_root,
        limit=limit,
        fusion_mode=fusion_mode,
        analysis_builder=gate_summary_record,
        disable_tqdm=disable_tqdm,
        progress=progress,
        print_components=print_components,
        device=device,
    )
    output_dir = Path(output_root) / "predictions" / dataset_name / f"fusion_{fusion_mode}" / str(seed)
    save_predictions_and_metrics(output_dir, predictions, metrics)
    write_run_manifest(
        output_dir,
        build_run_manifest(
            suite_name=suite_name,
            run_kind="fusion",
            run_name=f"fusion_{fusion_mode}",
            dataset=dataset_name,
            seed=seed,
            config_path=config_path,
            split_file=_resolved_split_path(dataset_name, seed, split_file, output_root),
            requested_command=requested_command or current_command(),
            expected_active_logits_losses=[],
            expected_disabled_losses=[],
            expected_knowledge_mode="verified",
            expected_evidence_mode=f"fusion_{fusion_mode}",
            extra={"fusion_mode": fusion_mode},
        ),
    )
    append_metric_row(Path(output_root) / "metrics" / "fusion_comparison.csv", _fusion_metric_row(dataset_name, seed, fusion_mode, metrics))
    write_jsonl(Path(output_root) / "analysis" / "gate_summary.jsonl", analysis, compact_tensors=True)
    return metrics


@torch.no_grad()
def run_framework_variant(
    dataset_name: str,
    model_name: str,
    seed: int = 42,
    config_path: str = "configs/config.yaml",
    split_file: str | None = None,
    output_root: str = "result",
    limit: int | None = None,
    ablation: AblationConfig | None = None,
    knowledge_mode: str = "verified",
    fusion_mode: str = "task_aware_gate_verified",
    analysis_builder: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None,
    disable_tqdm: bool = False,
    progress: ProgressConfig | None = None,
    print_components: bool = False,
    device: str = "cpu",
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Run a pipeline variant on the test split and return predictions/metrics/analysis."""

    set_seed(seed)
    runtime_device = _resolve_requested_device(device)
    cfg = deepcopy(load_yaml(config_path))
    cfg.setdefault("runtime", {})
    cfg["runtime"]["device"] = str(runtime_device)
    dataset = MemeDataset(
        dataset_root=cfg.get("paths", {}).get("dataset_root", "dataset/source"),
        annotation_root=cfg.get("paths", {}).get("annotation_root", "dataset/annotation"),
        dataset_names=[dataset_name],
        keep_missing_images=True,
        limit=limit,
    )
    samples = [dict(sample, label=label_to_int(sample.get("raw_label"))) for sample in dataset if label_to_int(sample.get("raw_label")) is not None]
    splits = _load_or_create_splits(dataset_name, dataset, seed, cfg, split_file, output_root)
    test_samples = split_samples(samples, splits).get("test", [])
    pipeline = HarmfulMemePipeline(cfg).to(runtime_device).eval()
    if print_components:
        from module.runner import print_pipeline_components

        print_pipeline_components(pipeline)
    predictions: list[dict[str, Any]] = []
    analysis: list[dict[str, Any]] = []
    analysis_builder = analysis_builder or gate_summary_record
    desc = ablation.name if ablation else f"knowledge {knowledge_mode}"
    progress_config = progress or ProgressConfig(disable=True if disable_tqdm else None)
    for sample in progress_iter(
        test_samples,
        desc=desc,
        config=progress_config,
        position=2,
        leave=progress_config.leave_batch,
    ):
        outputs = execute_variant_pipeline(pipeline, sample, ablation=ablation, knowledge_mode=knowledge_mode, fusion_mode=fusion_mode)
        extra = {
            "ablation": ablation.name if ablation else None,
            "knowledge_mode": knowledge_mode,
            "fusion_mode": fusion_mode,
        }
        if ablation:
            contract = get_ablation_contract(ablation.name)
            extra["ablation_contract"] = contract.to_dict()
            extra["component_state"] = component_state_for_ablation(ablation.name)
        predictions.append(
            stage_outputs_to_prediction_record(
                sample,
                outputs,
                model_name=model_name,
                seed=seed,
                extra=extra,
            )
        )
        analysis.append(analysis_builder(sample, outputs))
    metrics = _combined_metrics(predictions)
    return predictions, metrics, analysis


def execute_variant_pipeline(
    pipeline: HarmfulMemePipeline,
    sample: dict[str, Any],
    ablation: AblationConfig | None = None,
    knowledge_mode: str = "verified",
    fusion_mode: str = "task_aware_gate_verified",
) -> dict[str, Any]:
    """Execute stages manually so experiment-level ablations can transform outputs."""

    ablation = ablation or AblationConfig()
    stage_a = pipeline.stage_a(sample)
    if ablation.remove_roi:
        _remove_roi(stage_a)
    if ablation.remove_incongruity:
        _remove_incongruity(stage_a)

    if ablation.disable_retrieval or knowledge_mode == "no_knowledge":
        stage_b = _empty_stage_b(stage_a)
        stage_c = _empty_stage_c(stage_a)
    else:
        stage_b = pipeline.stage_b(stage_a, sample.get("ocr_text_full", ""))
        if ablation.disable_context_generation or knowledge_mode == "retrieved_only":
            _remove_generated_candidates(stage_b)
        if knowledge_mode == "generated_only":
            _keep_generated_candidates(stage_b)
        if knowledge_mode == "generated_retrieved":
            stage_c = _minimal_stage_c_from_stage_b(stage_a, stage_b)
        else:
            old_min = pipeline.stage_c.min_relevance
            if ablation.disable_relevance_scorer:
                pipeline.stage_c.min_relevance = 0.0
            stage_c = pipeline.stage_c(stage_a, stage_b)
            pipeline.stage_c.min_relevance = old_min
        if ablation.disable_support_verifier:
            _neutralize_support(stage_c)
        if ablation.disable_temporal_cultural_validator:
            _neutralize_validity(stage_c)

    stage_d = pipeline.stage_d(stage_a, stage_c)
    _apply_fusion_mode(pipeline, stage_d, fusion_mode)
    if ablation.disable_task_aware_gate:
        _disable_task_aware_gate(stage_d, gate_mode="shared_gate_ablation")
    elif fusion_mode in {"shared_gate", "concat_mlp", "mean_pooling", "cross_attention"}:
        _disable_task_aware_gate(stage_d, gate_mode=fusion_mode)
    stage_e = pipeline.stage_e(stage_a, stage_c, stage_d)
    if ablation.label_only_no_evidence:
        _label_only(stage_e)
    return {"stage_a": stage_a, "stage_b": stage_b, "stage_c": stage_c, "stage_d": stage_d, "stage_e": stage_e}


def gate_summary_record(sample: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
    stage_d = outputs["stage_d"]
    stage_e = outputs["stage_e"]
    task_gate = stage_d.gates.get("task_level")
    return {
        "sample_id": sample.get("sample_id"),
        "dataset_name": sample.get("dataset_name"),
        "target_gate": float(task_gate[0]) if isinstance(task_gate, torch.Tensor) and task_gate.numel() >= 3 else None,
        "intent_gate": float(task_gate[1]) if isinstance(task_gate, torch.Tensor) and task_gate.numel() >= 3 else None,
        "tactic_gate": float(task_gate[2]) if isinstance(task_gate, torch.Tensor) and task_gate.numel() >= 3 else None,
        "sample_gate": float(stage_d.gates.get("sample_level")) if isinstance(stage_d.gates.get("sample_level"), torch.Tensor) else None,
        "top_internal_evidence": stage_e.supporting_evidence.get("internal", [])[:3],
        "top_external_evidence": stage_e.supporting_evidence.get("external", [])[:3],
    }


def knowledge_analysis_record(sample: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
    """Return a compact sample-level knowledge comparison analysis row."""

    stage_b = outputs["stage_b"]
    stage_c = outputs["stage_c"]
    stage_e = outputs["stage_e"]
    verified_ids = {item.knowledge_id for item in stage_c.verified_items}
    retrieved = [candidate for candidate in stage_b.knowledge_candidates if candidate.candidate_type != "generated_hypothesis"]
    generated = [candidate for candidate in stage_b.knowledge_candidates if candidate.candidate_type == "generated_hypothesis"]
    verified = [_candidate_summary(item) for item in stage_c.verified_items]
    rejected = [_candidate_summary(candidate) for candidate in stage_b.knowledge_candidates if candidate.candidate_id not in verified_ids]
    fallback_count = sum(1 for item in stage_c.verified_items if item.source == "fallback")
    accepted_count = len(stage_c.verified_items)
    return {
        "sample_id": sample.get("sample_id"),
        "dataset_name": sample.get("dataset_name"),
        "generated_hypotheses": list(stage_b.generated_hypotheses),
        "generated_candidates": [_candidate_summary(candidate) for candidate in generated],
        "retrieved_candidates": [_candidate_summary(candidate) for candidate in retrieved],
        "verified_candidates": verified,
        "rejected_candidates": rejected,
        "accepted_knowledge_count": accepted_count,
        "fallback_knowledge_ratio": fallback_count / max(1, accepted_count),
        "final_prediction": stage_e.structured_prediction.get("harmfulness", {}),
    }


def append_metric_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _combined_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    harmfulness = compute_harmfulness_metrics(
        [record["gold_label"] for record in predictions if record.get("gold_label") is not None],
        [record["pred_label"] for record in predictions if record.get("gold_label") is not None],
        [record["prob_harmful"] for record in predictions if record.get("gold_label") is not None],
    )
    structured = evaluate_structured_predictions(predictions)
    return {**harmfulness, **structured}


def _load_or_create_splits(dataset_name, dataset, seed, cfg, split_file, output_root):
    if split_file:
        return load_split_file(split_file)
    split_path = Path(output_root) / "splits" / dataset_name / f"seed_{seed}.json"
    if split_path.exists():
        return load_split_file(split_path)
    splits = build_splits_for_dataset(dataset_name, dataset, seed=seed, dataset_root=cfg.get("paths", {}).get("dataset_root", "dataset/source"))
    save_splits(splits, dataset_name, seed, Path(output_root) / "splits")
    return splits


def _empty_stage_b(stage_a):
    device = stage_a.internal_tokens.device
    hidden_dim = stage_a.internal_tokens.size(-1)
    return StageBOutput(
        stage_a.sample_id,
        stage_a.dataset_name,
        QueryBundle(""),
        [],
        [],
        torch.zeros(0, hidden_dim, device=device),
        [],
        StageBMetadata(
            0,
            0,
            0,
            0,
            retrieval_stats={"retrieval_enabled": False, "context_generation_enabled": False},
            query_types={},
        ),
    )


def _empty_stage_c(stage_a):
    device = stage_a.internal_tokens.device
    hidden_dim = stage_a.internal_tokens.size(-1)
    return StageCOutput(
        stage_a.sample_id,
        stage_a.dataset_name,
        [],
        torch.zeros(0, hidden_dim, device=device),
        torch.zeros(0, 6, device=device),
        torch.zeros(0, device=device),
        "",
        StageCMetadata(0, 0, 8, 0.05),
    )


def _minimal_stage_c_from_stage_b(stage_a, stage_b):
    items = []
    from module.knowledge_filter_verifier import VerifiedKnowledgeItem

    for idx, candidate in enumerate(stage_b.knowledge_candidates):
        items.append(
            VerifiedKnowledgeItem(
                candidate.candidate_id,
                candidate.text,
                candidate.source,
                candidate.score,
                "insufficient",
                0.0,
                0.5,
                candidate.score,
                idx,
                {
                    "candidate": candidate.metadata,
                    "support_labels": {"target": "insufficient", "intent": "insufficient", "tactic": "insufficient"},
                    "claim_support": {"target": 0.0, "intent": 0.0, "tactic": 0.0},
                    "final_score_components": {"relevance": candidate.score, "validity": 0.5, "support": 0.0},
                },
            )
        )
    tokens = stage_b.candidate_tokens
    device = stage_b.candidate_tokens.device
    scores = torch.tensor([item.final_score for item in items], dtype=torch.float32, device=device)
    matrix = torch.zeros(len(items), 6, device=device)
    if len(items):
        matrix[:, 0] = scores
        matrix[:, 4] = 0.5
        matrix[:, 5] = scores
    return StageCOutput(
        stage_a.sample_id,
        stage_a.dataset_name,
        items,
        tokens,
        matrix,
        scores,
        "",
        StageCMetadata(
            len(items),
            len(items),
            8,
            0.0,
            claim_types=["target", "intent", "tactic"],
            score_fields=["relevance", "target_support", "intent_support", "tactic_support", "validity", "final"],
            support_matrix_columns=["relevance", "target_support", "intent_support", "tactic_support", "validity", "final"],
        ),
    )


def _remove_roi(stage_a) -> None:
    stage_a.roi_tokens = torch.zeros(0, stage_a.internal_tokens.size(-1), device=stage_a.internal_tokens.device)
    stage_a.evidence_items = [item for item in stage_a.evidence_items if item.evidence_type != "local_symbol"]
    stage_a.metadata.roi_count = 0


def _remove_incongruity(stage_a) -> None:
    for item in stage_a.evidence_items:
        if item.evidence_type == "cross_modal_incongruity":
            item.score = 0.0
            if item.token_index < stage_a.internal_tokens.size(0):
                stage_a.internal_tokens[item.token_index] = 0
    stage_a.auxiliary_scores["knowledge_need"] = 0.0
    stage_a.auxiliary_scores["incongruity_score"] = 0.0


def _remove_generated_candidates(stage_b) -> None:
    kept = [candidate for candidate in stage_b.knowledge_candidates if candidate.candidate_type != "generated_hypothesis"]
    _reset_candidates(stage_b, kept)
    stage_b.generated_hypotheses = []
    stage_b.metadata.generated_count = 0
    stage_b.metadata.retrieval_stats["context_generation_enabled"] = False


def _keep_generated_candidates(stage_b) -> None:
    kept = [candidate for candidate in stage_b.knowledge_candidates if candidate.candidate_type == "generated_hypothesis"]
    _reset_candidates(stage_b, kept)
    stage_b.metadata.retrieved_count = 0
    stage_b.metadata.retrieval_stats["retrieved_candidates_kept"] = False


def _reset_candidates(stage_b, kept) -> None:
    indices = [candidate.token_index for candidate in kept if 0 <= candidate.token_index < stage_b.candidate_tokens.size(0)]
    device = stage_b.candidate_tokens.device
    hidden_dim = stage_b.candidate_tokens.size(-1) if stage_b.candidate_tokens.dim() == 2 else 256
    stage_b.knowledge_candidates = kept
    stage_b.candidate_tokens = stage_b.candidate_tokens[indices] if indices else torch.zeros(0, hidden_dim, device=device)
    for idx, candidate in enumerate(stage_b.knowledge_candidates):
        candidate.token_index = idx


def _neutralize_support(stage_c) -> None:
    if stage_c.support_matrix.numel():
        stage_c.support_matrix[:, 1:4] = 0.0
    for item in stage_c.verified_items:
        item.support_label = "insufficient"
        item.support_score = 0.0
        item.metadata["support_labels"] = {"target": "insufficient", "intent": "insufficient", "tactic": "insufficient"}


def _neutralize_validity(stage_c) -> None:
    if stage_c.support_matrix.numel():
        stage_c.support_matrix[:, 4] = 1.0
    for item in stage_c.verified_items:
        item.validity_score = 1.0
        item.metadata["validity_components"] = {"credibility": 1.0, "temporal": 1.0, "cultural": 1.0}


def _apply_fusion_mode(pipeline: HarmfulMemePipeline, stage_d, fusion_mode: str) -> None:
    if fusion_mode == "mean_pooling":
        pooled = stage_d.internal_memory.mean(dim=0, keepdim=True).repeat(stage_d.fused_tokens.size(0), 1)
        stage_d.fused_tokens = pooled
    elif fusion_mode == "concat_mlp":
        stage_d.fused_tokens = 0.5 * (stage_d.internal_memory + stage_d.fused_tokens)
    elif fusion_mode == "cross_attention":
        pass
    elif fusion_mode in {"task_aware_gate", "task_aware_gate_verified", "shared_gate"}:
        pass
    else:
        return
    task_gate = stage_d.gates.get("task_level")
    if isinstance(task_gate, torch.Tensor) and task_gate.numel() >= 3:
        shared, latents = pipeline.stage_d.reasoning(stage_d.fused_tokens, task_gate, head_level=stage_d.gates.get("head_level"))
        stage_d.shared_reasoning_state = shared
        stage_d.task_latents = latents
    stage_d.metadata.gate_mode = fusion_mode


def _disable_task_aware_gate(stage_d, gate_mode: str = "shared_gate_ablation") -> None:
    task_gate = stage_d.gates.get("task_level")
    if isinstance(task_gate, torch.Tensor) and task_gate.numel():
        shared = task_gate.mean().repeat(3)
        stage_d.gates["task_level"] = shared
        for key in ["target", "intent", "tactic"]:
            stage_d.task_latents[key] = stage_d.shared_reasoning_state * shared.mean()
    stage_d.metadata.gate_mode = gate_mode
    stage_d.metadata.analysis_hooks["task_aware_gate_enabled"] = 0.0


def _label_only(stage_e) -> None:
    stage_e.supporting_evidence = {"internal": [], "external": []}
    stage_e.rationale = ""
    stage_e.structured_prediction["supporting_evidence"] = stage_e.supporting_evidence
    stage_e.structured_prediction["rationale"] = ""


def _resolved_split_path(dataset_name: str, seed: int, split_file: str | None, output_root: str) -> Path:
    if split_file:
        return Path(split_file)
    return Path(output_root) / "splits" / dataset_name / f"seed_{seed}.json"


def _ablation_metric_row(dataset, seed, ablation, metrics):
    return {
        "dataset": dataset,
        "seed": seed,
        "ablation": ablation,
        "harmfulness_macro_f1": metrics.get("macro_f1") or metrics.get("harmfulness_macro_f1"),
        "target_macro_f1": metrics.get("target_granularity_macro_f1"),
        "intent_macro_f1": metrics.get("intent_primary_macro_f1"),
        "tactic_macro_f1": metrics.get("tactic_multimodal_relation_macro_f1"),
        "delta_vs_full": "",
    }


def _fusion_metric_row(dataset, seed, fusion_mode, metrics):
    row = _ablation_metric_row(dataset, seed, fusion_mode, metrics)
    row.pop("ablation")
    row.pop("delta_vs_full")
    row["fusion_mode"] = fusion_mode
    return row


def _candidate_summary(candidate) -> dict[str, Any]:
    return {
        "id": getattr(candidate, "candidate_id", getattr(candidate, "knowledge_id", "")),
        "text": getattr(candidate, "text", ""),
        "source": getattr(candidate, "source", ""),
        "score": float(getattr(candidate, "score", getattr(candidate, "final_score", 0.0))),
        "type": getattr(candidate, "candidate_type", "verified"),
        "query": getattr(candidate, "query", ""),
    }


def _resolve_requested_device(device: str) -> torch.device:
    runtime_device = torch.device(device)
    if runtime_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device was requested for diagnostic ablation run, but CUDA is not available: {device}")
    return runtime_device
