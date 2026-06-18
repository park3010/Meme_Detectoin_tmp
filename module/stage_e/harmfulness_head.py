"""Harmfulness prediction head."""

from __future__ import annotations

import torch
from torch import nn


class HarmfulnessHead(nn.Module):
    """Binary harmfulness head with neural logits and interpretable lexical priors."""

    labels = ["non_harmful", "harmful"]

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, len(self.labels))

    def compute_logits(self, shared_state: torch.Tensor, text: str = "") -> torch.Tensor:
        """Return trainable logits with lightweight lexical priors added."""

        logits = self.classifier(shared_state.float())
        lowered = text.lower()
        harmful_cues = ["hate", "kill", "stupid", "inferior", "threat", "shit", "trash", "blame", "low on"]
        supportive_cues = ["support", "applause", "solidarity", "thanks"]
        if any(cue in lowered for cue in harmful_cues):
            logits[1] += 0.9
        if any(cue in lowered for cue in supportive_cues):
            logits[0] += 0.5
        return logits

    def forward(self, shared_state: torch.Tensor, text: str = "") -> dict[str, float]:
        logits = self.compute_logits(shared_state, text)
        probs = torch.softmax(logits, dim=-1).detach()
        return {label: float(probs[idx]) for idx, label in enumerate(self.labels)}
