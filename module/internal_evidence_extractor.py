"""Stage A: internal evidence extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from module.backbone.text import TextEncoderWrapper
from module.backbone.vision import CLIPWrapper, Detection, DetectorAdapter
from utils.tensor_utils import hashed_vector
from utils.text_utils import keyword_candidates, normalize_text, rhetorical_cues, sentence_chunks, target_presence_score


# =============================================================================
# Stage A schemas
# =============================================================================

@dataclass
class StageAInput:
    """Input container for internal evidence extraction."""

    sample_id: str
    dataset_name: str
    image_path: str | None
    ocr_text_full: str
    raw_label: int | str | None = None
    annotation: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_sample(cls, sample: dict[str, Any]) -> "StageAInput":
        """Build Stage A input from a dataset sample dictionary."""

        return cls(
            sample_id=str(sample.get("sample_id", "")),
            dataset_name=str(sample.get("dataset_name", "")),
            image_path=sample.get("image_path"),
            ocr_text_full=str(sample.get("ocr_text_full") or sample.get("ocr_text") or ""),
            raw_label=sample.get("raw_label"),
            annotation=sample.get("annotation"),
            metadata=dict(sample.get("metadata") or {}),
        )


@dataclass
class InternalEvidenceItem:
    """One internal evidence token and its provenance."""

    evidence_id: str
    evidence_type: str
    text: str
    score: float
    token_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageAMetadata:
    """Auxiliary weak-cue metadata, not final structured predictions."""

    image_backend: str
    text_backend: str
    roi_count: int
    token_count: int
    tensor_shapes: dict[str, list[int]] = field(default_factory=dict)
    auxiliary_labels: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class StageAOutput:
    """Internal evidence bank and weak cues consumed by later stages."""

    sample_id: str
    dataset_name: str
    internal_tokens: torch.Tensor
    evidence_items: list[InternalEvidenceItem]
    global_visual: torch.Tensor
    global_text: torch.Tensor
    patch_tokens: torch.Tensor
    text_tokens: torch.Tensor
    token_strings: list[str]
    roi_tokens: torch.Tensor
    auxiliary_scores: dict[str, float]
    metadata: StageAMetadata


# =============================================================================
# Visual/text/local evidence helpers
# =============================================================================

class GlobalVisualEncoder(nn.Module):
    """Encode image-level and coarse patch-level visual evidence."""

    def __init__(
        self,
        hidden_dim: int = 256,
        prefer_pretrained: bool = False,
        model_name: str = "ViT-B-32",
        device: str = "cpu",
        pretrained_tag: str | None = None,
        checkpoint_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = True,
        allow_download: bool = False,
        asset_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.clip = CLIPWrapper(
            hidden_dim=hidden_dim,
            prefer_pretrained=prefer_pretrained,
            model_name=model_name,
            device=device,
            pretrained_tag=pretrained_tag,
            checkpoint_path=checkpoint_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            allow_download=allow_download,
            asset_mode=asset_mode,
        )

    def forward(self, image_path: str | Path | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return global image embedding and four coarse patch tokens."""

        global_embedding = self.clip.encode_image(image_path)
        patch_tokens = []
        for idx in range(4):
            patch_hint = hashed_vector(f"patch:{idx}:{image_path}", dim=self.hidden_dim).to(global_embedding.device)
            patch_tokens.append(torch.nn.functional.normalize(0.75 * global_embedding + 0.25 * patch_hint, dim=0))
        return global_embedding, torch.stack(patch_tokens, dim=0)

class TextSemanticEncoder(nn.Module):
    """Encode OCR text into global and token-level semantic states."""

    def __init__(
        self,
        hidden_dim: int = 256,
        prefer_transformers: bool = False,
        model_name: str = "microsoft/deberta-v3-base",
        max_tokens: int = 64,
        device: str = "cpu",
        checkpoint_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        tokenizer_use_fast: bool = False,
        tokenizer_backend_policy: str | None = None,
        local_files_only: bool = True,
        allow_download: bool = False,
        asset_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.encoder = TextEncoderWrapper(
            hidden_dim=hidden_dim,
            prefer_transformers=prefer_transformers,
            model_name=model_name,
            max_tokens=max_tokens,
            device=device,
            checkpoint_path=checkpoint_path,
            cache_dir=cache_dir,
            tokenizer_use_fast=tokenizer_use_fast,
            tokenizer_backend_policy=tokenizer_backend_policy,
            local_files_only=local_files_only,
            allow_download=allow_download,
            asset_mode=asset_mode,
        )

    def forward(self, text: str) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        """Return global text embedding, token embeddings, and token strings."""

        return self.encoder.encode(text)

class LocalObjectSymbolExtractor(nn.Module):
    """Extract region-level evidence with a detector adapter and ROI encoder."""

    def __init__(
        self,
        hidden_dim: int = 256,
        detector_mode: str = "heuristic",
        max_rois: int = 3,
        prefer_pretrained_clip: bool = False,
        device: str = "cpu",
        clip_model_name: str = "ViT-B-32",
        pretrained_tag: str | None = None,
        checkpoint_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = True,
        allow_download: bool = False,
        asset_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.detector = DetectorAdapter(mode=detector_mode, max_rois=max_rois)
        self.roi_encoder = CLIPWrapper(
            hidden_dim=hidden_dim,
            prefer_pretrained=prefer_pretrained_clip,
            model_name=clip_model_name,
            device=device,
            pretrained_tag=pretrained_tag,
            checkpoint_path=checkpoint_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            allow_download=allow_download,
            asset_mode=asset_mode,
        )

    def forward(self, image_path: str | Path | None, ocr_text: str) -> tuple[list[Detection], torch.Tensor, list[dict[str, Any]]]:
        """Return detections, ROI tokens, and serializable metadata."""

        detections = self.detector.detect(image_path, ocr_text)
        boxes = [detection.box for detection in detections]
        tokens = self.roi_encoder.encode_rois(image_path, boxes)
        metadata = [
            {
                "box": list(detection.box),
                "label": detection.label,
                "score": detection.score,
                "metadata": detection.metadata,
            }
            for detection in detections
        ]
        return detections, tokens, metadata


# =============================================================================
# Cross-modal incongruity analysis
# =============================================================================

class CrossModalIncongruityAnalyzer(nn.Module):
    """Estimate visual-text alignment with bidirectional attention."""

    relation_labels = ["text_only", "image_only", "complementary", "incongruent", "cross_modal_implication"]

    def __init__(self, hidden_dim: int = 256, num_heads: int = 4) -> None:
        super().__init__()
        self.visual_to_text = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.0, batch_first=True)
        self.text_to_visual = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.0, batch_first=True)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.relation_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(self.relation_labels)),
        )

    def forward(
        self,
        visual: torch.Tensor,
        text: torch.Tensor,
        visual_tokens: torch.Tensor | None = None,
        text_tokens: torch.Tensor | None = None,
        ocr_text: str = "",
    ) -> tuple[torch.Tensor, dict[str, float], dict[str, object]]:
        """Return incongruity token, scalar scores, and interpretable labels."""

        visual = F.normalize(visual, dim=0)
        text = F.normalize(text, dim=0)

        visual_bank = _ensure_bank(visual, visual_tokens)
        text_bank = _ensure_bank(text, text_tokens)
        vt_context, vt_weights = self.visual_to_text(visual_bank.unsqueeze(0), text_bank.unsqueeze(0), text_bank.unsqueeze(0))
        tv_context, tv_weights = self.text_to_visual(text_bank.unsqueeze(0), visual_bank.unsqueeze(0), visual_bank.unsqueeze(0))
        vt_summary = vt_context.squeeze(0).mean(dim=0)
        tv_summary = tv_context.squeeze(0).mean(dim=0)

        similarity = torch.dot(visual, text).clamp(-1.0, 1.0)
        sim_matrix = F.normalize(visual_bank, dim=-1) @ F.normalize(text_bank, dim=-1).T
        token_alignment = sim_matrix.max(dim=1).values.mean().clamp(-1.0, 1.0)
        incongruity_tensor = (1.0 - (0.55 * similarity + 0.45 * token_alignment)).clamp(0.0, 2.0) / 2.0

        cue_scores = rhetorical_cues(ocr_text)
        target_score = target_presence_score(ocr_text)
        rhetorical_strength = max(cue_scores.values()) if cue_scores else 0.0
        knowledge_need = _knowledge_need(float(incongruity_tensor.detach()), target_score, rhetorical_strength, ocr_text)

        feature = torch.cat([visual, text, torch.abs(visual - text), visual * text], dim=0)
        relation_features = torch.cat(
            [
                feature,
                torch.tensor(
                    [
                        float(similarity.detach()),
                        float(token_alignment.detach()),
                        float(incongruity_tensor.detach()),
                        float(knowledge_need),
                        float(target_score),
                        float(rhetorical_strength),
                    ],
                    dtype=feature.dtype,
                    device=feature.device,
                ),
            ],
            dim=0,
        )
        relation_logits = self.relation_classifier(relation_features)
        relation_probs = torch.softmax(relation_logits, dim=-1)
        relation_idx = int(torch.argmax(relation_probs).item())
        relation_label = self.relation_labels[relation_idx]

        features = torch.cat(
            [
                visual,
                text,
                vt_summary,
                tv_summary,
                torch.abs(visual - text),
                visual * text,
                torch.abs(vt_summary - tv_summary),
                vt_summary * tv_summary,
            ],
            dim=0,
        )
        token = F.normalize(self.projection(features), dim=0)
        scores = {
            "visual_text_similarity": float(similarity.detach()),
            "token_alignment": float(token_alignment.detach()),
            "incongruity_score": float(incongruity_tensor.detach()),
            "knowledge_need": float(knowledge_need),
            "target_presence": float(target_score),
            "rhetorical_cues": float(rhetorical_strength),
            "multimodal_relation_confidence": float(relation_probs[relation_idx].detach()),
            "multimodal_relation_id": float(relation_idx),
        }
        labels = {
            "multimodal_relation": relation_label,
            "rhetorical_cues": cue_scores,
            "attention": {
                "visual_to_text_shape": list(vt_weights.squeeze(0).shape),
                "text_to_visual_shape": list(tv_weights.squeeze(0).shape),
            },
        }
        return token, scores, labels


def _ensure_bank(global_token: torch.Tensor, tokens: torch.Tensor | None) -> torch.Tensor:
    if tokens is None or tokens.numel() == 0:
        return global_token.unsqueeze(0)
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    tokens = tokens.to(device=global_token.device, dtype=global_token.dtype)
    return torch.cat([global_token.unsqueeze(0), tokens], dim=0)


def _knowledge_need(incongruity: float, target_score: float, rhetorical_strength: float, text: str) -> float:
    background_terms = {"who", "where", "why", "event", "history", "context", "brexit", "covid", "virus", "election"}
    lower_tokens = set(text.lower().split())
    background_bonus = 0.2 if lower_tokens & background_terms else 0.0
    length_bonus = 0.1 if len(text.split()) > 18 else 0.0
    return max(0.0, min(1.0, 0.45 * incongruity + 0.25 * target_score + 0.2 * rhetorical_strength + background_bonus + length_bonus))


# =============================================================================
# Stage A provenance and contract helpers
# =============================================================================



# =============================================================================
# InternalEvidenceExtractor
# =============================================================================

class InternalEvidenceExtractor(nn.Module):
    """Build the internal evidence token bank from image, OCR, ROIs, and alignment cues."""

    def __init__(
        self,
        hidden_dim: int = 256,
        max_text_tokens: int = 64,
        max_rois: int = 3,
        prefer_pretrained_clip: bool = False,
        prefer_transformers: bool = False,
        clip_model_name: str = "ViT-B-32",
        text_model_name: str = "microsoft/deberta-v3-base",
        detector_mode: str = "heuristic",
        device: str = "cpu",
        clip_pretrained_tag: str | None = None,
        clip_checkpoint_path: str | Path | None = None,
        clip_cache_dir: str | Path | None = None,
        clip_local_files_only: bool = True,
        clip_allow_download: bool = False,
        text_checkpoint_path: str | Path | None = None,
        text_cache_dir: str | Path | None = None,
        text_tokenizer_use_fast: bool = False,
        text_tokenizer_backend_policy: str | None = None,
        text_local_files_only: bool = True,
        text_allow_download: bool = False,
        clip_asset_mode: str | None = None,
        text_asset_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.visual_encoder = GlobalVisualEncoder(
            hidden_dim,
            prefer_pretrained_clip,
            model_name=clip_model_name,
            device=device,
            pretrained_tag=clip_pretrained_tag,
            checkpoint_path=clip_checkpoint_path,
            cache_dir=clip_cache_dir,
            local_files_only=clip_local_files_only,
            allow_download=clip_allow_download,
            asset_mode=clip_asset_mode,
        )
        self.text_encoder = TextSemanticEncoder(
            hidden_dim,
            prefer_transformers,
            model_name=text_model_name,
            max_tokens=max_text_tokens,
            device=device,
            checkpoint_path=text_checkpoint_path,
            cache_dir=text_cache_dir,
            tokenizer_use_fast=text_tokenizer_use_fast,
            tokenizer_backend_policy=text_tokenizer_backend_policy,
            local_files_only=text_local_files_only,
            allow_download=text_allow_download,
            asset_mode=text_asset_mode,
        )
        self.local_extractor = LocalObjectSymbolExtractor(
            hidden_dim=hidden_dim,
            detector_mode=detector_mode,
            max_rois=max_rois,
            prefer_pretrained_clip=prefer_pretrained_clip,
            device=device,
            clip_model_name=clip_model_name,
            pretrained_tag=clip_pretrained_tag,
            checkpoint_path=clip_checkpoint_path,
            cache_dir=clip_cache_dir,
            local_files_only=clip_local_files_only,
            allow_download=clip_allow_download,
            asset_mode=clip_asset_mode,
        )
        self.incongruity = CrossModalIncongruityAnalyzer(hidden_dim)

    def forward(self, sample: StageAInput | dict[str, Any]) -> StageAOutput:
        """Run Stage A for a single sample."""

        stage_input = sample if isinstance(sample, StageAInput) else StageAInput.from_sample(sample)
        visual_global, patch_tokens = self.visual_encoder(stage_input.image_path)
        text_global, text_tokens, token_strings = self.text_encoder(normalize_text(stage_input.ocr_text_full))
        detections, roi_tokens, roi_metadata = self.local_extractor(stage_input.image_path, stage_input.ocr_text_full)
        compute_device = _module_device(self.incongruity)
        visual_global = visual_global.to(compute_device)
        patch_tokens = patch_tokens.to(compute_device)
        text_global = text_global.to(compute_device)
        text_tokens = text_tokens.to(compute_device)
        roi_tokens = roi_tokens.to(compute_device)
        visual_bank = torch.cat([patch_tokens, roi_tokens], dim=0) if roi_tokens.numel() else patch_tokens
        incongruity_token, aux_scores, aux_labels = self.incongruity(
            visual_global,
            text_global,
            visual_tokens=visual_bank,
            text_tokens=text_tokens,
            ocr_text=stage_input.ocr_text_full,
        )
        # Stage A scores and relation labels are weak internal cues. Stage E is
        # responsible for final structured target/intent/tactic predictions.
        aux_scores = dict(aux_scores)
        aux_scores["knowledge_need_score"] = aux_scores["knowledge_need"]
        aux_scores["target_presence_score"] = aux_scores["target_presence"]
        aux_labels = dict(aux_labels)
        aux_labels["stage_a_multimodal_relation"] = aux_labels.get("multimodal_relation", "unknown")

        tokens = [visual_global, text_global]
        evidence_items = [
            InternalEvidenceItem(
                evidence_id=f"{stage_input.sample_id}:visual_global",
                evidence_type="global_visual",
                text="Global visual meme context.",
                score=1.0,
                token_index=0,
                metadata=_evidence_metadata(
                    "global_visual",
                    {"image_path": stage_input.image_path, "shape": list(visual_global.shape)},
                ),
            ),
            InternalEvidenceItem(
                evidence_id=f"{stage_input.sample_id}:text_global",
                evidence_type="global_text",
                text=stage_input.ocr_text_full,
                score=1.0,
                token_index=1,
                metadata=_evidence_metadata(
                    "global_text",
                    {
                        "top_keywords": keyword_candidates(stage_input.ocr_text_full, limit=8),
                        "rhetorical_cues": rhetorical_cues(stage_input.ocr_text_full),
                        "shape": list(text_global.shape),
                    },
                ),
            ),
        ]

        text_chunks = sentence_chunks(stage_input.ocr_text_full, limit=3)
        for chunk_idx, chunk in enumerate(text_chunks):
            token_index = len(tokens)
            chunk_hint = hashed_vector(chunk, dim=self.hidden_dim).to(text_global.device)
            chunk_token = F.normalize(0.65 * text_global + 0.35 * chunk_hint, dim=0)
            tokens.append(chunk_token)
            evidence_items.append(
                InternalEvidenceItem(
                    evidence_id=f"{stage_input.sample_id}:text_span:{chunk_idx}",
                    evidence_type="text_span",
                    text=chunk,
                    score=max(0.45, 0.8 - chunk_idx * 0.1),
                    token_index=token_index,
                    metadata=_evidence_metadata(
                        "text_span",
                        {"keywords": keyword_candidates(chunk, limit=5)},
                    ),
                )
            )

        for patch_idx, patch_token in enumerate(patch_tokens[:2]):
            token_index = len(tokens)
            tokens.append(patch_token)
            evidence_items.append(
                InternalEvidenceItem(
                    evidence_id=f"{stage_input.sample_id}:visual_patch:{patch_idx}",
                    evidence_type="visual_patch",
                    text=f"Coarse visual patch {patch_idx}.",
                    score=0.55,
                    token_index=token_index,
                    metadata=_evidence_metadata("visual_patch", {"patch_index": patch_idx}),
                )
            )

        for roi_idx, detection in enumerate(detections):
            token_index = len(tokens)
            if roi_idx < roi_tokens.size(0):
                tokens.append(roi_tokens[roi_idx])
            else:
                tokens.append(torch.zeros(self.hidden_dim, device=visual_global.device))
            evidence_items.append(
                InternalEvidenceItem(
                    evidence_id=f"{stage_input.sample_id}:roi:{roi_idx}",
                    evidence_type="local_symbol",
                    text=detection.label,
                    score=float(detection.score),
                    token_index=token_index,
                    metadata=_evidence_metadata("local_symbol", roi_metadata[roi_idx]),
                )
            )

        token_index = len(tokens)
        tokens.append(incongruity_token)
        evidence_items.append(
            InternalEvidenceItem(
                evidence_id=f"{stage_input.sample_id}:incongruity",
                evidence_type="cross_modal_incongruity",
                text=f"Visual-text relation: {aux_labels.get('multimodal_relation', 'unknown')}.",
                score=float(aux_scores["incongruity_score"]),
                token_index=token_index,
                metadata=_evidence_metadata(
                    "cross_modal_incongruity",
                    {**dict(aux_scores), **aux_labels},
                ),
            )
        )
        internal_tokens = F.normalize(torch.stack(tokens, dim=0), dim=-1)
        metadata = StageAMetadata(
            image_backend=self.visual_encoder.clip.backend,
            text_backend=self.text_encoder.encoder.backend,
            roi_count=len(detections),
            token_count=internal_tokens.size(0),
            tensor_shapes={
                "internal_tokens": list(internal_tokens.shape),
                "patch_tokens": list(patch_tokens.shape),
                "text_tokens": list(text_tokens.shape),
                "roi_tokens": list(roi_tokens.shape),
            },
            auxiliary_labels=aux_labels,
            notes=["phase-2 multimodal evidence bank with attention-based incongruity"],
        )
        return StageAOutput(
            sample_id=stage_input.sample_id,
            dataset_name=stage_input.dataset_name,
            internal_tokens=internal_tokens,
            evidence_items=evidence_items,
            global_visual=visual_global,
            global_text=text_global,
            patch_tokens=patch_tokens,
            text_tokens=text_tokens,
            token_strings=token_strings,
            roi_tokens=roi_tokens,
            auxiliary_scores=aux_scores,
            metadata=metadata,
        )


_EVIDENCE_CONTRACT = {
    "global_visual": {"modality": "image", "grounding_type": "global", "is_heuristic": False},
    "global_text": {"modality": "text", "grounding_type": "global", "is_heuristic": False},
    "text_span": {"modality": "text", "grounding_type": "span", "is_heuristic": False},
    "visual_patch": {"modality": "image", "grounding_type": "patch", "is_heuristic": True},
    "local_symbol": {"modality": "image_text", "grounding_type": "pseudo_roi", "is_heuristic": True},
    "cross_modal_incongruity": {"modality": "cross_modal", "grounding_type": "relation", "is_heuristic": True},
}


def _module_device(module: nn.Module) -> torch.device:
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


def _evidence_metadata(evidence_type: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge stable Stage A evidence provenance with existing metadata."""

    contract = _EVIDENCE_CONTRACT.get(
        evidence_type,
        {"modality": "unknown", "grounding_type": "internal", "is_heuristic": True},
    )
    return {
        **dict(extra or {}),
        **contract,
        "source_stage": "stage_a",
    }


# =============================================================================
# Public exports
# =============================================================================

__all__ = [
    "StageAInput",
    "InternalEvidenceItem",
    "StageAMetadata",
    "StageAOutput",
    "GlobalVisualEncoder",
    "TextSemanticEncoder",
    "LocalObjectSymbolExtractor",
    "CrossModalIncongruityAnalyzer",
    "InternalEvidenceExtractor",
]
