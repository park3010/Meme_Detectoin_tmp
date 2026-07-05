"""Pipeline model, inference helpers, and artifact runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from dataset import MemeDataset
from module.evidence_fusion_reasoning import EvidenceFusionReasoning
from module.external_knowledge_acquisition import ExternalKnowledgeAcquisition, QueryBundle, StageBMetadata, StageBOutput
from module.internal_evidence_extractor import InternalEvidenceExtractor
from module.knowledge_filter_verifier import KnowledgeRelevanceFilterVerifier, StageCMetadata, StageCOutput
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

    def forward(self, sample: dict[str, Any], run_until: str = "e", *, ablation: Any | None = None) -> dict[str, Any]:
        """Run the pipeline through the requested final stage."""

        normalized_until = run_until.lower().replace("stage_", "")
        outputs: dict[str, Any] = {}
        stage_a = self.stage_a(sample)
        if _flag(ablation, "disable_roi", "remove_roi"):
            _remove_roi(stage_a)
        if _flag(ablation, "disable_incongruity", "remove_incongruity"):
            _remove_incongruity(stage_a)
        outputs["stage_a"] = stage_a
        if normalized_until == "a":
            return outputs

        if _flag(ablation, "disable_retrieval"):
            stage_b = _empty_stage_b(stage_a)
        else:
            stage_b = self.stage_b(stage_a, sample.get("ocr_text_full", ""))
            if _flag(ablation, "disable_context_generation"):
                _remove_generated_candidates(stage_b)
        outputs["stage_b"] = stage_b
        if normalized_until == "b":
            return outputs

        if _flag(ablation, "disable_retrieval"):
            stage_c = _empty_stage_c(stage_a)
        else:
            old_min = self.stage_c.min_relevance
            if _flag(ablation, "disable_relevance_scorer"):
                self.stage_c.min_relevance = 0.0
            stage_c = self.stage_c(stage_a, stage_b)
            self.stage_c.min_relevance = old_min
            if _flag(ablation, "disable_support_verifier"):
                _neutralize_support(stage_c)
            if _flag(ablation, "disable_temporal_cultural_validator"):
                _neutralize_validity(stage_c)
        outputs["stage_c"] = stage_c
        if normalized_until == "c":
            return outputs

        stage_d = self.stage_d(stage_a, stage_c, disable_task_aware_gate=_flag(ablation, "disable_task_aware_gate"))
        outputs["stage_d"] = stage_d
        if normalized_until == "d":
            return outputs

        outputs["stage_e"] = self.stage_e(stage_a, stage_c, stage_d)
        if _flag(ablation, "label_only_no_evidence"):
            _label_only(outputs["stage_e"])
        return outputs


def _flag(ablation: Any | None, *names: str) -> bool:
    if ablation is None:
        return False
    for name in names:
        if bool(getattr(ablation, name, False)):
            return True
    return False


def _empty_stage_b(stage_a: Any) -> StageBOutput:
    device = stage_a.internal_tokens.device
    hidden_dim = int(stage_a.internal_tokens.size(-1))
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
            retrieval_stats={
                "retrieval_enabled": False,
                "context_generation_enabled": False,
                "ablation_runtime": "train_time",
            },
            query_types={},
        ),
    )


def _empty_stage_c(stage_a: Any) -> StageCOutput:
    device = stage_a.internal_tokens.device
    hidden_dim = int(stage_a.internal_tokens.size(-1))
    return StageCOutput(
        stage_a.sample_id,
        stage_a.dataset_name,
        [],
        torch.zeros(0, hidden_dim, device=device),
        torch.zeros(0, 6, device=device),
        torch.zeros(0, device=device),
        "",
        StageCMetadata(
            0,
            0,
            8,
            0.05,
            claim_types=["target", "intent", "tactic"],
            score_fields=["relevance", "target_support", "intent_support", "tactic_support", "validity", "final"],
            support_matrix_columns=["relevance", "target_support", "intent_support", "tactic_support", "validity", "final"],
            verification_policy={"retrieval_disabled": True, "train_time_ablation": True},
        ),
    )


def _remove_roi(stage_a: Any) -> None:
    device = stage_a.internal_tokens.device
    hidden_dim = int(stage_a.internal_tokens.size(-1))
    stage_a.roi_tokens = torch.zeros(0, hidden_dim, device=device)
    stage_a.evidence_items = [item for item in stage_a.evidence_items if item.evidence_type != "local_symbol"]
    stage_a.metadata.roi_count = 0
    stage_a.metadata.notes.append("ablation: roi/local_symbol evidence removed")


def _remove_incongruity(stage_a: Any) -> None:
    for item in stage_a.evidence_items:
        if item.evidence_type == "cross_modal_incongruity":
            item.score = 0.0
            if item.token_index < stage_a.internal_tokens.size(0):
                stage_a.internal_tokens[item.token_index] = 0
    stage_a.auxiliary_scores["knowledge_need"] = 0.0
    stage_a.auxiliary_scores["knowledge_need_score"] = 0.0
    stage_a.auxiliary_scores["incongruity_score"] = 0.0
    stage_a.metadata.notes.append("ablation: cross-modal incongruity neutralized")


def _remove_generated_candidates(stage_b: StageBOutput) -> None:
    kept = [candidate for candidate in stage_b.knowledge_candidates if candidate.candidate_type != "generated_hypothesis"]
    _reset_candidates(stage_b, kept)
    stage_b.generated_hypotheses = []
    stage_b.metadata.generated_count = 0
    stage_b.metadata.retrieval_stats["context_generation_enabled"] = False
    stage_b.metadata.retrieval_stats["context_generation_ablation"] = True


def _reset_candidates(stage_b: StageBOutput, kept: list[Any]) -> None:
    device = stage_b.candidate_tokens.device
    hidden_dim = int(stage_b.candidate_tokens.size(-1)) if stage_b.candidate_tokens.dim() == 2 else 256
    indices = [candidate.token_index for candidate in kept if 0 <= candidate.token_index < stage_b.candidate_tokens.size(0)]
    stage_b.knowledge_candidates = kept
    stage_b.candidate_tokens = stage_b.candidate_tokens[indices] if indices else torch.zeros(0, hidden_dim, device=device)
    for idx, candidate in enumerate(stage_b.knowledge_candidates):
        candidate.token_index = idx


def _neutralize_support(stage_c: StageCOutput) -> None:
    if stage_c.support_matrix.numel():
        stage_c.support_matrix[:, 1:4] = 0.0
        support_free = (stage_c.support_matrix[:, 0] * stage_c.support_matrix[:, 4]).clamp(0.0, 1.0)
        stage_c.support_matrix[:, 5] = support_free
        stage_c.final_scores = support_free
    for item in stage_c.verified_items:
        item.support_label = "insufficient"
        item.support_score = 0.0
        item.final_score = float(stage_c.final_scores[item.token_index]) if 0 <= item.token_index < stage_c.final_scores.numel() else item.final_score
        item.metadata["support_labels"] = {"target": "insufficient", "intent": "insufficient", "tactic": "insufficient"}
        item.metadata["claim_support"] = {"target": 0.0, "intent": 0.0, "tactic": 0.0}
        components = dict(item.metadata.get("final_score_components") or {})
        components["support"] = 0.0
        item.metadata["final_score_components"] = components
    stage_c.metadata.verification_policy["support_verifier_enabled"] = False
    stage_c.metadata.verification_policy["support_verifier_ablation"] = "train_time_neutralized"


def _neutralize_validity(stage_c: StageCOutput) -> None:
    if stage_c.support_matrix.numel():
        stage_c.support_matrix[:, 4] = 1.0
        stage_c.final_scores = stage_c.support_matrix[:, 5]
    for item in stage_c.verified_items:
        item.validity_score = 1.0
        item.metadata["validity_components"] = {"credibility": 1.0, "temporal": 1.0, "cultural": 1.0}
    stage_c.metadata.verification_policy["validity_enabled"] = False
    stage_c.metadata.verification_policy["validity_ablation"] = "train_time_neutralized"


def _label_only(stage_e: Any) -> None:
    stage_e.supporting_evidence = {"internal": [], "external": []}
    stage_e.rationale = ""
    stage_e.structured_prediction["supporting_evidence"] = stage_e.supporting_evidence
    stage_e.structured_prediction["rationale"] = ""


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
