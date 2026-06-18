"""Evidence gating modules for Stage D."""

from __future__ import annotations

import torch
from torch import nn


class EvidenceGate(nn.Module):
    """Compute token-, sample-, and task-level gates."""

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.token_gate = nn.Linear(hidden_dim, 1)
        self.compare_token_gate = nn.Linear(hidden_dim * 4, 1)
        self.sample_gate = nn.Linear(hidden_dim, 1)
        self.knowledge_need_gate = nn.Linear(1, 1)
        self.task_gate = nn.Linear(hidden_dim, 3)
        self.head_gate = nn.Linear(hidden_dim, 3)

    def forward(
        self,
        fused_tokens: torch.Tensor,
        internal_memory: torch.Tensor | None = None,
        knowledge_need: float | torch.Tensor | None = None,
        q_int: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Apply gates and return gated tokens plus gate tensors."""

        if internal_memory is None:
            internal_memory = fused_tokens
        if internal_memory.shape != fused_tokens.shape:
            raise ValueError(f"internal_memory shape {tuple(internal_memory.shape)} must match fused_tokens {tuple(fused_tokens.shape)}")

        compare = torch.cat([internal_memory, fused_tokens, torch.abs(fused_tokens - internal_memory), fused_tokens * internal_memory], dim=-1)
        token_level = torch.sigmoid(0.5 * self.token_gate(fused_tokens).squeeze(-1) + 0.5 * self.compare_token_gate(compare).squeeze(-1))
        summary = q_int if q_int is not None else fused_tokens.mean(dim=0)
        learned_sample = torch.sigmoid(self.sample_gate(summary)).squeeze()
        need_tensor = _knowledge_need_tensor(knowledge_need, fused_tokens.device, fused_tokens.dtype)
        need_gate = torch.sigmoid(self.knowledge_need_gate(need_tensor.view(1))).squeeze()
        sample_level = (0.65 * learned_sample + 0.35 * need_tensor.squeeze()).clamp(0.0, 1.0) * (0.5 + 0.5 * need_gate)
        task_level = torch.sigmoid(self.task_gate(summary))
        head_level = torch.sigmoid(self.head_gate(fused_tokens)).transpose(0, 1)
        external_gate = (token_level * sample_level).clamp(0.0, 1.0)
        gated = external_gate.unsqueeze(-1) * fused_tokens + (1.0 - external_gate).unsqueeze(-1) * internal_memory
        return gated, {
            "token_level": token_level,
            "sample_level": sample_level,
            "knowledge_need": need_tensor.squeeze(),
            "external_gate": external_gate,
            "task_level": task_level,
            "head_level": head_level,
        }


def _knowledge_need_tensor(value: float | torch.Tensor | None, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype).flatten()[0].clamp(0.0, 1.0)
    if value is None:
        return torch.tensor(0.5, device=device, dtype=dtype)
    return torch.tensor(float(value), device=device, dtype=dtype).clamp(0.0, 1.0)
