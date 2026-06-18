"""Intent prediction head."""

from __future__ import annotations

import torch
from torch import nn


class IntentHead(nn.Module):
    """Predict primary communicative intent."""

    labels = [
        "ridicule_mockery",
        "denigration_insult",
        "harassment_threat",
        "persuasion_propaganda",
        "misinformation_deception",
        "mobilization_incitement",
        "solidarity_support",
        "self_expression",
        "entertainment",
        "criticism",
        "neutral",
        "other",
    ]

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, len(self.labels))

    def compute_logits(self, intent_latent: torch.Tensor, text: str = "") -> torch.Tensor:
        """Return trainable intent logits with fallback lexical priors."""

        logits = self.classifier(intent_latent.float())
        lowered = text.lower()
        if any(term in lowered for term in ["shit", "stupid", "laugh", "lol", "joke"]):
            _add_prior(logits, self.labels, "ridicule_mockery", 0.6)
        if any(term in lowered for term in ["vote", "propaganda", "change", "america"]):
            _add_prior(logits, self.labels, "persuasion_propaganda", 0.25)
        if any(term in lowered for term in ["support", "applause", "thanks"]):
            _add_prior(logits, self.labels, "solidarity_support", 0.6)
        if any(term in lowered for term in ["join", "march", "fight back", "rise", "mobilize"]):
            _add_prior(logits, self.labels, "mobilization_incitement", 0.35)
        if any(term in lowered for term in ["criticize", "critique", "wrong", "failed", "corrupt"]):
            _add_prior(logits, self.labels, "criticism", 0.3)
        return logits

    def forward(self, intent_latent: torch.Tensor, text: str = "") -> dict[str, float]:
        logits = self.compute_logits(intent_latent, text)
        probs = torch.softmax(logits, dim=-1).detach()
        return {label: float(probs[idx]) for idx, label in enumerate(self.labels)}


def _add_prior(logits: torch.Tensor, labels: list[str], label: str, amount: float) -> None:
    """Safely add a lexical prior when the label exists."""

    if label in labels:
        logits[labels.index(label)] += amount
