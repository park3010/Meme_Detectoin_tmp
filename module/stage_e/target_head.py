"""Target prediction head."""

from __future__ import annotations

import torch
from torch import nn

from utils.text_utils import capitalized_spans


class TargetHead(nn.Module):
    """Predict target granularity/presence."""

    labels = ["none", "individual", "organization", "community", "society"]

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, len(self.labels))

    def compute_logits(self, target_latent: torch.Tensor, text: str = "") -> torch.Tensor:
        """Return trainable target granularity logits with fallback priors."""

        logits = self.classifier(target_latent.float())
        lowered = text.lower()
        if capitalized_spans(text, limit=2):
            logits[self.labels.index("individual")] += 0.35
        if any(term in lowered for term in ["people", "women", "men", "thais", "european", "america"]):
            logits[self.labels.index("community")] += 0.45
        if any(term in lowered for term in ["government", "school", "party", "media"]):
            logits[self.labels.index("organization")] += 0.35
        return logits

    def forward(self, target_latent: torch.Tensor, text: str = "") -> dict[str, float]:
        logits = self.compute_logits(target_latent, text)
        probs = torch.softmax(logits, dim=-1).detach()
        return {label: float(probs[idx]) for idx, label in enumerate(self.labels)}
