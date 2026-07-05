"""Pipeline model, inference helpers, and artifact runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from dataset import MemeDataset
from module.evidence_fusion_reasoning import EvidenceFusionReasoning
from module.external_knowledge_acquisition import ExternalKnowledgeAcquisition
from module.internal_evidence_extractor import InternalEvidenceExtractor
from module.knowledge_filter_verifier import KnowledgeRelevanceFilterVerifier
from module.structured_interpretation_head import StructuredInterpretationHead
from utils.io import deep_update, load_yaml, write_json, write_jsonl
from utils.logging_utils import setup_logger
from utils.seed import set_seed
from utils.tensor_utils import tensor_to_python


logger = setup_logger(__name__)


def _section(config: dict[str, Any], *names: str) -> dict[str, Any]:
    for name in names:
        value = config.get(name)
        if isinstance(value, dict):
            return value
    return {}


def _stage_config(config: dict[str, Any], stage_name: str) -> dict[str, Any]:
    stages = config.get("stages", {})
    if isinstance(stages, dict) and isinstance(stages.get(stage_name), dict):
        return stages[stage_name]
    value = config.get(stage_name)
    return value if isinstance(value, dict) else {}


# =============================================================================
# Pipeline model
# =============================================================================


class HarmfulMemePipeline(nn.Module):
    """End-to-end harmful meme interpretation pipeline."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        config = config or {}
        model_cfg = config.get("model", {})
        backbone_cfg = _section(config, "backbone", "backbones")
        runtime_cfg = config.get("runtime", {})
        paths_cfg = config.get("paths", {})
        hidden_dim = int(model_cfg.get("hidden_dim", 256))
        device = str(runtime_cfg.get("device", "cpu"))

        clip_cfg = backbone_cfg.get("clip", {})
        text_cfg = backbone_cfg.get("text", {})
        detector_cfg = backbone_cfg.get("detector", {})
        retriever_cfg = backbone_cfg.get("retriever", {})
        stage_a_cfg = _stage_config(config, "stage_a")
        stage_c_cfg = _stage_config(config, "stage_c")
        stage_d_cfg = _stage_config(config, "stage_d")
        self.stage_a = InternalEvidenceExtractor(
            hidden_dim=hidden_dim,
            max_rois=int(stage_a_cfg.get("max_rois", 3)),
            prefer_pretrained_clip=bool(clip_cfg.get("prefer_pretrained", False)),
            prefer_transformers=bool(text_cfg.get("prefer_transformers", False)),
            clip_model_name=str(clip_cfg.get("model_name", "ViT-B-32")),
            text_model_name=str(text_cfg.get("model_name", "microsoft/deberta-v3-base")),
            detector_mode=str(detector_cfg.get("mode", "heuristic")),
            device=device,
            clip_pretrained_tag=clip_cfg.get("pretrained_tag"),
            clip_checkpoint_path=clip_cfg.get("checkpoint_path"),
            clip_cache_dir=clip_cfg.get("cache_dir"),
            clip_local_files_only=bool(clip_cfg.get("local_files_only", True)),
            clip_allow_download=bool(clip_cfg.get("allow_download", False)),
            clip_asset_mode=clip_cfg.get("asset_mode"),
            text_checkpoint_path=text_cfg.get("checkpoint_path"),
            text_cache_dir=text_cfg.get("cache_dir"),
            text_tokenizer_use_fast=bool(text_cfg.get("tokenizer_use_fast", False)),
            text_tokenizer_backend_policy=text_cfg.get("tokenizer_backend_policy"),
            text_local_files_only=bool(text_cfg.get("local_files_only", True)),
            text_allow_download=bool(text_cfg.get("allow_download", False)),
            text_asset_mode=text_cfg.get("asset_mode"),
        )
        self.stage_b = ExternalKnowledgeAcquisition(
            hidden_dim=hidden_dim,
            corpus_paths=list(paths_cfg.get("retrieval_corpus_paths", []) or []),
            top_k=int(model_cfg.get("knowledge_top_k", 8)),
            fallback_candidates=bool(retriever_cfg.get("fallback_candidates", True)),
            prefer_transformers=bool(text_cfg.get("prefer_transformers", False)),
            text_model_name=str(text_cfg.get("model_name", "microsoft/deberta-v3-base")),
            text_checkpoint_path=text_cfg.get("checkpoint_path"),
            text_cache_dir=text_cfg.get("cache_dir"),
            text_tokenizer_use_fast=bool(text_cfg.get("tokenizer_use_fast", False)),
            text_tokenizer_backend_policy=text_cfg.get("tokenizer_backend_policy"),
            text_local_files_only=bool(text_cfg.get("local_files_only", True)),
            text_allow_download=bool(text_cfg.get("allow_download", False)),
            text_asset_mode=text_cfg.get("asset_mode"),
            max_documents=retriever_cfg.get("max_documents"),
            use_cross_encoder_rerank=bool(retriever_cfg.get("use_cross_encoder_rerank", True)),
            device=device,
        )
        self.stage_c = KnowledgeRelevanceFilterVerifier(
            hidden_dim=hidden_dim,
            top_k=int(model_cfg.get("knowledge_top_k", 8)),
            min_relevance=float(stage_c_cfg.get("min_relevance", 0.05)),
            allow_low_relevance_fallback=bool(stage_c_cfg.get("allow_low_relevance_fallback", True)),
        )
        self.stage_d = EvidenceFusionReasoning(
            hidden_dim=hidden_dim,
            num_heads=int(stage_d_cfg.get("num_heads", 4)),
            num_layers=int(stage_d_cfg.get("num_layers", 2)),
        )
        self.stage_e = StructuredInterpretationHead(hidden_dim=hidden_dim)

    def forward(self, sample: dict[str, Any], run_until: str = "e") -> dict[str, Any]:
        """Run the pipeline through the requested final stage."""

        normalized_until = run_until.lower().replace("stage_", "")
        outputs: dict[str, Any] = {}
        stage_a = self.stage_a(sample)
        outputs["stage_a"] = stage_a
        if normalized_until == "a":
            return outputs

        stage_b = self.stage_b(stage_a, sample.get("ocr_text_full", ""))
        outputs["stage_b"] = stage_b
        if normalized_until == "b":
            return outputs

        stage_c = self.stage_c(stage_a, stage_b)
        outputs["stage_c"] = stage_c
        if normalized_until == "c":
            return outputs

        stage_d = self.stage_d(stage_a, stage_c)
        outputs["stage_d"] = stage_d
        if normalized_until == "d":
            return outputs

        outputs["stage_e"] = self.stage_e(stage_a, stage_c, stage_d)
        return outputs


# =============================================================================
# Inference helpers
# =============================================================================


def run_single_sample(sample: dict[str, Any], config_path: str | Path = "configs/config.yaml", run_until: str = "e") -> dict[str, Any]:
    """Run one in-memory sample through the pipeline."""

    runner = PipelineRunner(config_path=config_path)
    return runner.pipeline(sample, run_until=run_until)


# =============================================================================
# Runtime / artifact runner
# =============================================================================

logger = setup_logger(__name__)


class PipelineRunner:
    """Load config/data, execute stages, and save outputs."""

    def __init__(self, config_path: str | Path = "configs/config.yaml", overrides: dict[str, Any] | None = None) -> None:
        self.config_path = Path(config_path)
        self.config = load_yaml(self.config_path)
        if overrides:
            self.config = deep_update(self.config, overrides)
        set_seed(int(self.config.get("runtime", {}).get("seed", 13)))
        self.pipeline = HarmfulMemePipeline(self.config).eval()
        self.result_root = Path(self.config.get("paths", {}).get("result_root", "result"))

    def load_dataset(self, dataset_names: list[str] | None = None, limit: int | None = None) -> MemeDataset:
        """Load configured dataset."""

        runtime_limit = self.config.get("runtime", {}).get("max_samples")
        effective_limit = limit if limit is not None else runtime_limit
        return MemeDataset(
            dataset_root=self.config.get("paths", {}).get("dataset_root", "dataset/source"),
            annotation_root=self.config.get("paths", {}).get("annotation_root", "dataset/annotation"),
            dataset_names=dataset_names,
            keep_missing_images=True,
            limit=int(effective_limit) if effective_limit else None,
        )

    @torch.no_grad()
    def run(
        self,
        dataset_names: list[str] | None = None,
        limit: int | None = None,
        run_until: str = "e",
        save: bool = True,
    ) -> list[dict[str, Any]]:
        """Run the configured pipeline and optionally save stage outputs."""

        dataset = self.load_dataset(dataset_names=dataset_names, limit=limit)
        logger.info("Running pipeline on %s samples through Stage %s", len(dataset), run_until.upper())
        records: list[dict[str, Any]] = []
        for sample in dataset:
            outputs = self.pipeline(sample, run_until=run_until)
            records.append({"sample": sample, "outputs": outputs})
        if save:
            self.save_outputs(records, run_until=run_until)
        return records

    def save_outputs(self, records: list[dict[str, Any]], run_until: str = "e") -> None:
        """Save readable per-stage artifacts under result/."""

        stage_order = ["a", "b", "c", "d", "e"]
        active = set(stage_order[: stage_order.index(run_until.lower()) + 1])
        if "a" in active:
            write_jsonl(
                self.result_root / "stage_a" / "internal_evidence.jsonl",
                [record["outputs"]["stage_a"] for record in records if "stage_a" in record["outputs"]],
            )
        if "b" in active:
            write_jsonl(
                self.result_root / "stage_b" / "knowledge_candidates.jsonl",
                [record["outputs"]["stage_b"] for record in records if "stage_b" in record["outputs"]],
            )
        if "c" in active:
            write_jsonl(
                self.result_root / "stage_c" / "verified_knowledge.jsonl",
                [record["outputs"]["stage_c"] for record in records if "stage_c" in record["outputs"]],
            )
        if "d" in active:
            self._save_stage_d_pt(records)
        if "e" in active:
            write_jsonl(
                self.result_root / "stage_e" / "final_predictions.jsonl",
                [record["outputs"]["stage_e"].structured_prediction for record in records if "stage_e" in record["outputs"]],
                compact_tensors=True,
            )
            self._save_analysis_exports(records)
        if self.config.get("runtime", {}).get("save_per_sample", False):
            for record in records:
                sample_id = record["sample"]["sample_id"]
                write_json(self.result_root / "samples" / f"{sample_id}.json", record)

    def _save_stage_d_pt(self, records: list[dict[str, Any]]) -> None:
        path = self.result_root / "stage_d" / "fusion_states.pt"
        legacy_path = self.result_root / "stage_d" / "fusion_outputs.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = []
        for record in records:
            output = record["outputs"].get("stage_d")
            if output is None:
                continue
            payload.append(
                {
                    "sample_id": output.sample_id,
                    "dataset_name": output.dataset_name,
                    "shared_reasoning_state": output.shared_reasoning_state.detach().cpu(),
                    "internal_memory": output.internal_memory.detach().cpu(),
                    "fused_tokens": output.fused_tokens.detach().cpu(),
                    "cross_attention_weights": output.cross_attention_weights.detach().cpu(),
                    "task_latents": {key: value.detach().cpu() for key, value in output.task_latents.items()},
                    "gates": tensor_to_python(output.gates),
                    "metadata": tensor_to_python(output.metadata),
                }
            )
        torch.save(payload, path)
        torch.save(payload, legacy_path)
        logger.info("Saved Stage D tensor artifact to %s", path)

    def _save_analysis_exports(self, records: list[dict[str, Any]]) -> None:
        attribution_records = []
        summary_records = []
        for record in records:
            stage_e = record["outputs"].get("stage_e")
            if stage_e is None:
                continue
            attribution_records.append(
                {
                    "sample_id": stage_e.sample_id,
                    "dataset_name": stage_e.dataset_name,
                    "supporting_evidence": stage_e.supporting_evidence,
                }
            )
            structured = stage_e.structured_prediction
            summary_records.append(
                {
                    "sample_id": stage_e.sample_id,
                    "dataset_name": stage_e.dataset_name,
                    "harmfulness": structured.get("harmfulness", {}).get("label"),
                    "target": structured.get("target", {}).get("granularity"),
                    "intent": structured.get("intent", {}).get("primary"),
                    "tactic": structured.get("tactic", {}).get("label"),
                    "rationale": structured.get("rationale", ""),
                }
            )
        write_jsonl(self.result_root / "analysis" / "evidence_attribution.jsonl", attribution_records)
        write_jsonl(self.result_root / "analysis" / "sample_summaries.jsonl", summary_records)


# =============================================================================
# Public exports
# =============================================================================

__all__ = ["HarmfulMemePipeline", "PipelineRunner", "run_single_sample"]
