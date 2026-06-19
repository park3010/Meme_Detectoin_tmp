"""Tactic prediction head."""

from __future__ import annotations

import torch
from torch import nn


class TacticHead(nn.Module):
    """Predict rhetorical or multimodal tactic."""

    labels = [
        "sarcasm_irony",
        "stereotype",
        "slur",
        "smear",
        "exaggeration",
        "fear_appeal",
        "metaphor",
        "dehumanization",
        "objectification",
        "exclusion",
        "dog_whistle",
        "conspiracy_cue",
        "whataboutism",
        "slogan",
        "image_text_incongruity",
        "juxtaposition",
        "sexual_innuendo",
        "propaganda",
        "none",
        "other",
    ]
    relation_labels = [
        "complementary",
        "incongruent",
        "cross_modal_implication",
        "text_only",
        "image_only",
        "none",
        "other",
    ]

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, len(self.labels))
        self.relation_classifier = nn.Linear(hidden_dim, len(self.relation_labels))

    def compute_logits(self, tactic_latent: torch.Tensor, text: str = "") -> torch.Tensor:
        """Return trainable tactic logits with fallback rhetorical priors."""

        logits = self.classifier(tactic_latent.float())
        lowered = text.lower()
        if "?" in text or any(term in lowered for term in ["sure", "totally", "dankest", "then i told"]):
            _add_prior(logits, self.labels, "sarcasm_irony", 0.5)
        if any(term in lowered for term in ["all", "cultural difference", "real men"]):
            _add_prior(logits, self.labels, "stereotype", 0.35)
        if any(term in lowered for term in ["always", "never", "highly rated", "wreck"]):
            _add_prior(logits, self.labels, "exaggeration", 0.35)
        if any(term in lowered for term in ["danger", "threat", "invasion", "destroy", "fear"]):
            _add_prior(logits, self.labels, "fear_appeal", 0.3)
        if any(term in lowered for term in ["animal", "rats", "vermin", "parasite", "subhuman"]):
            _add_prior(logits, self.labels, "dehumanization", 0.3)
        if any(term in lowered for term in ["object", "property", "thing"]):
            _add_prior(logits, self.labels, "objectification", 0.25)
        if any(term in lowered for term in ["ban", "deport", "keep out", "not welcome"]):
            _add_prior(logits, self.labels, "exclusion", 0.3)
        if any(term in lowered for term in ["globalist", "replacement", "13/50"]):
            _add_prior(logits, self.labels, "dog_whistle", 0.3)
        if any(term in lowered for term in ["conspiracy", "hoax", "deep state", "they don't want you to know"]):
            _add_prior(logits, self.labels, "conspiracy_cue", 0.3)
        if any(term in lowered for term in ["what about", "but what about"]):
            _add_prior(logits, self.labels, "whataboutism", 0.3)
        if any(term in lowered for term in ["yeah right"]):
            _add_prior(logits, self.labels, "image_text_incongruity", 0.25)
        if any(term in lowered for term in ["vs", "versus", "before and after"]):
            _add_prior(logits, self.labels, "juxtaposition", 0.25)
        if any(term in lowered for term in ["sexy", "hot", "nude", "bed"]):
            _add_prior(logits, self.labels, "sexual_innuendo", 0.25)
        if any(term in lowered for term in ["vote", "agenda", "patriot", "enemy"]):
            _add_prior(logits, self.labels, "propaganda", 0.25)
        return logits

    def compute_relation_logits(
        self,
        tactic_latent: torch.Tensor,
        text: str = "",
        stage_a_relation: str = "",
    ) -> torch.Tensor:
        """Return trainable multimodal-relation logits with weak cue priors."""

        logits = self.relation_classifier(tactic_latent.float())
        relation = str(stage_a_relation).strip().lower().replace("-", "_").replace(" ", "_")
        if relation in self.relation_labels:
            _add_prior(logits, self.relation_labels, relation, 0.25)
        elif relation in {"mismatch", "image_text_incongruity", "contradictory"}:
            _add_prior(logits, self.relation_labels, "incongruent", 0.2)
        elif relation in {"cross_modal", "cross_modal_inference", "implicit_cross_modal"}:
            _add_prior(logits, self.relation_labels, "cross_modal_implication", 0.2)

        lowered = text.lower().strip()
        token_count = len(lowered.split())
        if not lowered:
            _add_prior(logits, self.relation_labels, "image_only", 0.15)
            _add_prior(logits, self.relation_labels, "none", 0.1)
        elif token_count >= 12:
            _add_prior(logits, self.relation_labels, "text_only", 0.12)
        if any(term in lowered for term in ["yeah right", "doesn't match", "but the image", "contradicts"]):
            _add_prior(logits, self.relation_labels, "incongruent", 0.15)
        if any(term in lowered for term in ["implies", "suggests", "context", "symbolizes"]):
            _add_prior(logits, self.relation_labels, "cross_modal_implication", 0.12)
        return logits

    def forward(self, tactic_latent: torch.Tensor, text: str = "") -> dict[str, float]:
        logits = self.compute_logits(tactic_latent, text)
        probs = torch.softmax(logits, dim=-1).detach()
        return {label: float(probs[idx]) for idx, label in enumerate(self.labels)}


def _add_prior(logits: torch.Tensor, labels: list[str], label: str, amount: float) -> None:
    """Safely add a lexical prior when the label exists."""

    if label in labels:
        logits[labels.index(label)] += amount
