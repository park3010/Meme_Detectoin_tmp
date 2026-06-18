"""Redundancy reduction for Stage C."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from module.stage_c.schemas import VerifiedKnowledgeItem
from utils.tensor_utils import hashed_vector
from utils.text_utils import jaccard_similarity


class RedundancyReducer:
    """Remove near-duplicate knowledge while preserving high scores."""

    def reduce(self, items: list[VerifiedKnowledgeItem], top_k: int = 8, duplicate_threshold: float = 0.82) -> list[VerifiedKnowledgeItem]:
        """Return a diverse top-k subset."""

        selected: list[VerifiedKnowledgeItem] = []
        for item in sorted(items, key=lambda value: value.final_score, reverse=True):
            if any(jaccard_similarity(item.text, chosen.text) >= duplicate_threshold for chosen in selected):
                continue
            selected.append(item)
            if len(selected) >= top_k:
                break
        return selected

    def reduce_with_tokens(
        self,
        items: list[VerifiedKnowledgeItem],
        tokens: dict[str, torch.Tensor],
        top_k: int = 8,
        cluster_threshold: float = 0.78,
    ) -> list[VerifiedKnowledgeItem]:
        """Select high-scoring items while discouraging one-cluster collapse."""

        selected: list[VerifiedKnowledgeItem] = []
        cluster_counts: list[int] = []
        centroids: list[torch.Tensor] = []
        for item in sorted(items, key=lambda value: value.final_score, reverse=True):
            token = tokens.get(item.knowledge_id, hashed_vector(item.text))
            token = F.normalize(token.float(), dim=0)
            cluster_idx = _assign_cluster(token, centroids, cluster_threshold)
            diversity_penalty = 0.08 * cluster_counts[cluster_idx] if cluster_idx is not None else 0.0
            adjusted_score = item.final_score - diversity_penalty
            if adjusted_score <= 0 and len(selected) >= max(1, top_k // 2):
                continue
            if cluster_idx is None:
                centroids.append(token)
                cluster_counts.append(1)
            else:
                cluster_counts[cluster_idx] += 1
                centroids[cluster_idx] = F.normalize(centroids[cluster_idx] * 0.7 + token * 0.3, dim=0)
            item.metadata.setdefault("redundancy", {})
            item.metadata["redundancy"].update({"cluster": cluster_idx if cluster_idx is not None else len(centroids) - 1, "adjusted_score": adjusted_score})
            selected.append(item)
            if len(selected) >= top_k:
                break
        return selected


def _assign_cluster(token: torch.Tensor, centroids: list[torch.Tensor], threshold: float) -> int | None:
    if not centroids:
        return None
    similarities = [float(torch.dot(token.detach(), centroid.detach())) for centroid in centroids]
    best_idx, best_score = max(enumerate(similarities), key=lambda pair: pair[1])
    return best_idx if best_score >= threshold else None
