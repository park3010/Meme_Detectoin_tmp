"""Knowledge-conditioned cross-attention for Stage D."""

from __future__ import annotations

import torch
from torch import nn


class KnowledgeConditionedCrossAttention(nn.Module):
    """Fuse internal memory with verified knowledge tokens."""

    def __init__(self, hidden_dim: int = 256, num_heads: int = 4) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.0, batch_first=True)
        self.support_projection = nn.Linear(1, hidden_dim)
        self.task_support_projection = nn.Linear(4, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, hidden_dim))

    def forward(
        self,
        internal_memory: torch.Tensor,
        knowledge_tokens: torch.Tensor,
        support_scores: torch.Tensor | None = None,
        task_support_matrix: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fused internal tokens and attention weights."""

        if knowledge_tokens.numel() == 0 or knowledge_tokens.size(0) == 0:
            weights = torch.zeros(internal_memory.size(0), 0, dtype=internal_memory.dtype, device=internal_memory.device)
            return internal_memory, weights
        query = internal_memory.unsqueeze(0)
        key_value_tokens = knowledge_tokens.to(device=internal_memory.device, dtype=internal_memory.dtype)
        if support_scores is not None and support_scores.numel() == key_value_tokens.size(0):
            support = support_scores.to(device=internal_memory.device, dtype=internal_memory.dtype).clamp(0.0, 1.0).unsqueeze(-1)
            if task_support_matrix is not None and task_support_matrix.numel() and task_support_matrix.size(0) == key_value_tokens.size(0):
                task_support = task_support_matrix.to(device=internal_memory.device, dtype=internal_memory.dtype).clamp(0.0, 1.0)
                key_value_tokens = key_value_tokens + self.task_support_projection(torch.cat([support, task_support], dim=-1))
            else:
                key_value_tokens = key_value_tokens + self.support_projection(support)
        key_value = key_value_tokens.unsqueeze(0)
        attended, weights = self.attention(query=query, key=key_value, value=key_value, need_weights=True)
        fused = self.norm(internal_memory + attended.squeeze(0))
        fused = self.norm(fused + self.ffn(fused))
        return fused, weights.squeeze(0)
