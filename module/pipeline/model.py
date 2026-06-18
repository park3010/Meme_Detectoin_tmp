"""Top-level model that wires stages A-E."""

from __future__ import annotations

from typing import Any

from torch import nn

from module.stage_a.extractor import InternalEvidenceExtractor
from module.stage_b.acquisition import ExternalKnowledgeAcquisition
from module.stage_c.verifier import KnowledgeRelevanceFilterVerifier
from module.stage_d.fusion import EvidenceFusionReasoning
from module.stage_e.interpretation import StructuredInterpretationHead


class HarmfulMemePipeline(nn.Module):
    """End-to-end harmful meme interpretation pipeline."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        config = config or {}
        model_cfg = config.get("model", {})
        backbone_cfg = config.get("backbones", {})
        runtime_cfg = config.get("runtime", {})
        paths_cfg = config.get("paths", {})
        hidden_dim = int(model_cfg.get("hidden_dim", 256))
        device = str(runtime_cfg.get("device", "cpu"))

        clip_cfg = backbone_cfg.get("clip", {})
        text_cfg = backbone_cfg.get("text", {})
        detector_cfg = backbone_cfg.get("detector", {})
        retriever_cfg = backbone_cfg.get("retriever", {})
        self.stage_a = InternalEvidenceExtractor(
            hidden_dim=hidden_dim,
            max_rois=int(config.get("stage_a", {}).get("max_rois", 3)),
            prefer_pretrained_clip=bool(clip_cfg.get("prefer_pretrained", False)),
            prefer_transformers=bool(text_cfg.get("prefer_transformers", False)),
            clip_model_name=str(clip_cfg.get("model_name", "ViT-B-32")),
            text_model_name=str(text_cfg.get("model_name", "microsoft/deberta-v3-base")),
            detector_mode=str(detector_cfg.get("mode", "heuristic")),
            device=device,
        )
        self.stage_b = ExternalKnowledgeAcquisition(
            hidden_dim=hidden_dim,
            corpus_paths=list(paths_cfg.get("retrieval_corpus_paths", []) or []),
            top_k=int(model_cfg.get("knowledge_top_k", 8)),
            fallback_candidates=bool(retriever_cfg.get("fallback_candidates", True)),
            prefer_transformers=bool(text_cfg.get("prefer_transformers", False)),
            text_model_name=str(text_cfg.get("model_name", "microsoft/deberta-v3-base")),
            max_documents=retriever_cfg.get("max_documents"),
            use_cross_encoder_rerank=bool(retriever_cfg.get("use_cross_encoder_rerank", True)),
            device=device,
        )
        self.stage_c = KnowledgeRelevanceFilterVerifier(
            hidden_dim=hidden_dim,
            top_k=int(model_cfg.get("knowledge_top_k", 8)),
            min_relevance=float(config.get("stage_c", {}).get("min_relevance", 0.05)),
            allow_low_relevance_fallback=bool(config.get("stage_c", {}).get("allow_low_relevance_fallback", True)),
        )
        self.stage_d = EvidenceFusionReasoning(
            hidden_dim=hidden_dim,
            num_heads=int(config.get("stage_d", {}).get("num_heads", 4)),
            num_layers=int(config.get("stage_d", {}).get("num_layers", 2)),
        )
        self.stage_e = StructuredInterpretationHead(hidden_dim=hidden_dim)

    def forward(self, sample: dict[str, Any], run_until: str = "e") -> dict[str, Any]:
        """Run the pipeline through the requested final stage."""

        outputs: dict[str, Any] = {}
        stage_a = self.stage_a(sample)
        outputs["stage_a"] = stage_a
        if run_until.lower() == "a":
            return outputs

        stage_b = self.stage_b(stage_a, sample.get("ocr_text_full", ""))
        outputs["stage_b"] = stage_b
        if run_until.lower() == "b":
            return outputs

        stage_c = self.stage_c(stage_a, stage_b)
        outputs["stage_c"] = stage_c
        if run_until.lower() == "c":
            return outputs

        stage_d = self.stage_d(stage_a, stage_c)
        outputs["stage_d"] = stage_d
        if run_until.lower() == "d":
            return outputs

        outputs["stage_e"] = self.stage_e(stage_a, stage_c, stage_d)
        return outputs
