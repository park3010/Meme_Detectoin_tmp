"""Cross-modal incongruity analysis for Stage A."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from utils.text_utils import rhetorical_cues, target_presence_score


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
