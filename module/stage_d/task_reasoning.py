"""Task-aware reasoning heads for Stage D."""

from __future__ import annotations

import torch
from torch import nn


class TaskAwareReasoningHeads(nn.Module):
    """Produce shared state and task-specific latent vectors."""

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.private_norm = nn.LayerNorm(hidden_dim)
        self.target = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.intent = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.tactic = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))

    def forward(
        self,
        gated_tokens: torch.Tensor,
        task_gates: torch.Tensor,
        head_level: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return shared reasoning state and task latents."""

        pooled = gated_tokens.mean(dim=0)
        shared_state = self.shared(pooled)
        gates = task_gates.flatten()
        private_pools = _private_pools(gated_tokens, head_level)
        latents = {
            "target": self.target(shared_state + self.private_norm(private_pools["target"])) * gates[0],
            "intent": self.intent(shared_state + self.private_norm(private_pools["intent"])) * gates[1],
            "tactic": self.tactic(shared_state + self.private_norm(private_pools["tactic"])) * gates[2],
        }
        return shared_state, latents


def _private_pools(gated_tokens: torch.Tensor, head_level: torch.Tensor | None) -> dict[str, torch.Tensor]:
    names = ["target", "intent", "tactic"]
    if head_level is None or head_level.numel() == 0:
        pooled = gated_tokens.mean(dim=0)
        return {name: pooled for name in names}
    pools: dict[str, torch.Tensor] = {}
    for idx, name in enumerate(names):
        weights = head_level[idx].float()
        weights = weights / weights.sum().clamp(min=1e-6)
        pools[name] = (gated_tokens * weights.unsqueeze(-1)).sum(dim=0)
    return pools
