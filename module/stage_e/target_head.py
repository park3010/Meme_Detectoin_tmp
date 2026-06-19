"""Target prediction head."""

from __future__ import annotations

import torch
from torch import nn

from utils.text_utils import capitalized_spans, target_presence_score


class TargetHead(nn.Module):
    """Predict target granularity/presence."""

    labels = ["none", "individual", "organization", "community", "society"]
    presence_labels = ["explicit", "implicit", "none"]

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, len(self.labels))
        self.presence_classifier = nn.Linear(hidden_dim, len(self.presence_labels))

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

    def compute_presence_logits(self, target_latent: torch.Tensor, text: str = "") -> torch.Tensor:
        """Return trainable target-presence logits with small lexical priors."""

        logits = self.presence_classifier(target_latent.float())
        presence_score = target_presence_score(text)
        entities = capitalized_spans(text, limit=2)
        if entities:
            logits[self.presence_labels.index("explicit")] += 0.25
        if not entities and 0.25 <= presence_score < 0.55:
            logits[self.presence_labels.index("implicit")] += 0.2
        if not entities and presence_score < 0.25:
            logits[self.presence_labels.index("none")] += 0.2
        return logits

    def forward(self, target_latent: torch.Tensor, text: str = "") -> dict[str, float]:
        logits = self.compute_logits(target_latent, text)
        probs = torch.softmax(logits, dim=-1).detach()
        return {label: float(probs[idx]) for idx, label in enumerate(self.labels)}
