"""Stage A orchestration module."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from module.stage_a.incongruity_analyzer import CrossModalIncongruityAnalyzer
from module.stage_a.local_symbol_extractor import LocalObjectSymbolExtractor
from module.stage_a.schemas import InternalEvidenceItem, StageAInput, StageAMetadata, StageAOutput
from module.stage_a.text_semantic_encoder import TextSemanticEncoder
from module.stage_a.visual_encoder import GlobalVisualEncoder
from utils.tensor_utils import hashed_vector
from utils.text_utils import keyword_candidates, normalize_text, rhetorical_cues, sentence_chunks


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
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.visual_encoder = GlobalVisualEncoder(hidden_dim, prefer_pretrained_clip, model_name=clip_model_name, device=device)
        self.text_encoder = TextSemanticEncoder(hidden_dim, prefer_transformers, model_name=text_model_name, max_tokens=max_text_tokens, device=device)
        self.local_extractor = LocalObjectSymbolExtractor(
            hidden_dim=hidden_dim,
            detector_mode=detector_mode,
            max_rois=max_rois,
            prefer_pretrained_clip=prefer_pretrained_clip,
            device=device,
        )
        self.incongruity = CrossModalIncongruityAnalyzer(hidden_dim)

    def forward(self, sample: StageAInput | dict[str, Any]) -> StageAOutput:
        """Run Stage A for a single sample."""

        stage_input = sample if isinstance(sample, StageAInput) else StageAInput.from_sample(sample)
        visual_global, patch_tokens = self.visual_encoder(stage_input.image_path)
        text_global, text_tokens, token_strings = self.text_encoder(normalize_text(stage_input.ocr_text_full))
        detections, roi_tokens, roi_metadata = self.local_extractor(stage_input.image_path, stage_input.ocr_text_full)
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
