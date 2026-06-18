"""Template-based rationale generation."""

from __future__ import annotations

from module.stage_e.schemas import Prediction


class TemplateRationaleGenerator:
    """Generate a concise evidence-grounded rationale.

    A constrained generator can be injected in Phase 3 as long as it exposes a
    `generate(payload: dict) -> str` method. The deterministic template remains
    the safe default for reproducible research runs.
    """

    def __init__(self, constrained_generator: object | None = None) -> None:
        self.constrained_generator = constrained_generator

    def generate(
        self,
        harmfulness: Prediction,
        target: Prediction,
        intent: Prediction,
        tactic: Prediction,
        evidence: dict[str, list[dict[str, object]]],
    ) -> str:
        """Return a deterministic rationale paragraph."""

        internal_text = "; ".join(str(item.get("text", "")) for item in evidence.get("internal", [])[:2])
        external_text = "; ".join(str(item.get("text", "")) for item in evidence.get("external", [])[:1])
        if self.constrained_generator is not None and hasattr(self.constrained_generator, "generate"):
            payload = {
                "harmfulness": harmfulness,
                "target": target,
                "intent": intent,
                "tactic": tactic,
                "internal_evidence": evidence.get("internal", []),
                "external_evidence": evidence.get("external", []),
            }
            return str(self.constrained_generator.generate(payload))
        parts = [
            f"The meme is predicted as {harmfulness.label} with likely target granularity '{target.label}'.",
            f"The main intent is {intent.label}, and the salient tactic is {tactic.label}.",
        ]
        if internal_text:
            parts.append(f"Internal evidence includes: {internal_text}.")
        if external_text:
            parts.append(f"External context considered: {external_text}.")
        return " ".join(parts)
