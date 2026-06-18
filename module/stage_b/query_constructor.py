"""Query construction for Stage B."""

from __future__ import annotations

from module.stage_a.schemas import StageAOutput
from module.stage_b.schemas import QueryBundle
from utils.text_utils import capitalized_spans, keyword_candidates, normalize_text, rhetorical_cues, sentence_chunks


class QueryConstructor:
    """Build diverse external-knowledge queries from OCR and internal cues."""

    def __init__(self, max_queries: int = 6) -> None:
        self.max_queries = max_queries

    def build(self, ocr_text: str, stage_a: StageAOutput) -> QueryBundle:
        """Create query bundle for one sample."""

        clean = normalize_text(ocr_text)
        keywords = keyword_candidates(clean, limit=8)
        entities = capitalized_spans(ocr_text, limit=4)
        chunks = sentence_chunks(clean, limit=3)
        cues = rhetorical_cues(clean)
        relation = stage_a.metadata.auxiliary_labels.get("multimodal_relation", "") if hasattr(stage_a.metadata, "auxiliary_labels") else ""
        target_like = [item.text for item in stage_a.evidence_items if item.evidence_type in {"text_span", "local_symbol"} and item.text]
        joined_keywords = " ".join(keywords[:6])
        return QueryBundle(
            ocr_query=clean[:280] if clean else joined_keywords,
            entity_queries=[f"{entity} identity context meme reference alias" for entity in entities],
            event_queries=[f"event background timeline {chunk}" for chunk in chunks if any(token in chunk.lower() for token in ["virus", "election", "debate", "brexit", "covid", "war", "school"])],
            meme_template_queries=[f"meme template rhetoric {relation} {' '.join(cues.keys())} {joined_keywords}".strip()]
            if joined_keywords or cues
            else [],
            social_context_queries=[f"social cultural context target group {joined_keywords}"] if joined_keywords else [],
            target_hypothesis_queries=[f"target intent tactic hypothesis {normalize_text(text)[:180]}" for text in target_like[:3]],
        )
