"""Evidence-aware relevance scoring for Stage C."""

from __future__ import annotations

import torch
from torch import nn

from module.backbones.cross_encoder_adapter import CrossEncoderAdapter
from module.stage_a.schemas import StageAOutput
from module.stage_b.schemas import KnowledgeCandidate
from utils.tensor_utils import hashed_vector
from utils.text_utils import normalize_text


class EvidenceAwareRelevanceScorer(nn.Module):
    """Score candidate knowledge with pairwise embedding features."""

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cross_encoder = CrossEncoderAdapter()
        self.feature_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def summarize_internal(self, stage_a: StageAOutput) -> str:
        """Create a compact textual summary of internal evidence."""

        parts = [
            normalize_text(item.text)
            for item in stage_a.evidence_items
            if item.evidence_type in {"global_text", "local_symbol", "cross_modal_incongruity"}
        ]
        return "; ".join(part for part in parts if part)[:600]

    def score(self, internal_summary: str, candidate: KnowledgeCandidate) -> float:
        """Return relevance score using lexical pair scoring and retriever prior."""

        pair_score = self.cross_encoder.score(internal_summary, candidate.text)
        prior = max(0.0, min(1.0, candidate.score))
        return 0.75 * pair_score + 0.25 * prior

    def score_pair(
        self,
        internal_vector: torch.Tensor,
        candidate_vector: torch.Tensor,
        internal_summary: str,
        candidate: KnowledgeCandidate,
    ) -> tuple[float, torch.Tensor, dict[str, float]]:
        """Score relevance from [q; k; q*k; |q-k|] plus lexical priors."""

        if candidate_vector.numel() == 0:
            candidate_vector = hashed_vector(candidate.text, dim=self.hidden_dim).to(internal_vector.device)
        else:
            candidate_vector = candidate_vector.to(internal_vector.device)
        q = torch.nn.functional.normalize(internal_vector.float(), dim=0)
        k = torch.nn.functional.normalize(candidate_vector.float(), dim=0)
        lexical = self.cross_encoder.score(internal_summary, candidate.text)
        retriever_prior = max(0.0, min(1.0, candidate.score))
        cosine = float(torch.dot(q, k).clamp(-1.0, 1.0).detach())
        feature = torch.cat(
            [
                q,
                k,
                q * k,
                torch.abs(q - k),
                torch.tensor([lexical, retriever_prior, cosine], dtype=q.dtype, device=q.device),
            ],
            dim=0,
        )
        neural = float(torch.sigmoid(self.feature_mlp(feature)).squeeze().detach())
        score = max(0.0, min(1.0, 0.4 * neural + 0.25 * lexical + 0.2 * retriever_prior + 0.15 * max(0.0, cosine)))
        return score, feature.detach(), {
            "neural_relevance": neural,
            "lexical_relevance": lexical,
            "retriever_prior": retriever_prior,
            "embedding_cosine": cosine,
        }
