"""Context hypothesis generation for Stage B."""

from __future__ import annotations

from module.backbones.generator_adapter import GeneratorAdapter
from module.stage_b.schemas import KnowledgeCandidate
from utils.text_utils import normalize_text


class ContextAugmentationGenerator:
    """Generate short candidate hypotheses from top evidence."""

    def __init__(self, max_items: int = 3) -> None:
        self.generator = GeneratorAdapter()
        self.max_items = max_items

    def generate(self, ocr_text: str, retrieved: list[KnowledgeCandidate], sample_id: str = "sample") -> tuple[list[str], list[KnowledgeCandidate]]:
        """Return hypothesis strings grounded in retrieved evidence."""

        evidence_summary = "; ".join(normalize_text(item.text) for item in retrieved[:3])
        hypotheses = self.generator.generate_hypotheses(ocr_text, evidence_summary=evidence_summary, max_items=self.max_items)
        candidates = [
            KnowledgeCandidate(
                candidate_id=f"{sample_id}:generated_hypothesis:{idx}",
                text=hypothesis,
                source="template_generator",
                score=max(0.2, 0.55 - idx * 0.08),
                candidate_type="generated_hypothesis",
                metadata={"evidence_summary": evidence_summary[:500], "grounded_in": [item.candidate_id for item in retrieved[:3]]},
            )
            for idx, hypothesis in enumerate(hypotheses)
        ]
        return hypotheses, candidates
