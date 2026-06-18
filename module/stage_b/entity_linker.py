"""Entity and concept linking for Stage B."""

from __future__ import annotations

from module.stage_b.schemas import LinkedEntity
from utils.text_utils import capitalized_spans, keyword_candidates, normalize_text


class EntityConceptLinker:
    """Link surface entities/concepts and attach alias expansion metadata."""

    alias_table = {
        "usa": ["united states", "america", "american"],
        "america": ["united states", "usa", "american"],
        "covid": ["coronavirus", "covid-19", "pandemic"],
        "corona": ["coronavirus", "covid-19"],
        "trump": ["donald trump", "us president"],
        "obama": ["barack obama", "us president"],
        "brexit": ["uk european union withdrawal", "britain eu"],
    }

    def link(self, text: str, surface_forms: list[str] | None = None) -> list[LinkedEntity]:
        """Return entity-like and concept-like links from text plus evidence surfaces."""

        links: list[LinkedEntity] = []
        seen: set[str] = set()
        for span in capitalized_spans(text, limit=8):
            norm = span.lower()
            if norm in seen:
                continue
            seen.add(norm)
            links.append(
                LinkedEntity(
                    surface=span,
                    normalized=norm,
                    link_type="entity",
                    confidence=0.72,
                    metadata={"aliases": self.alias_table.get(norm, [])},
                )
            )
        for surface in surface_forms or []:
            clean = normalize_text(surface)
            norm = clean.lower()
            if not clean or norm in seen:
                continue
            seen.add(norm)
            link_type = "visual_symbol" if any(prefix in norm for prefix in ["region_", "patch", "roi"]) else "evidence_surface"
            confidence = 0.62 if link_type == "visual_symbol" else 0.56
            links.append(
                LinkedEntity(
                    surface=clean,
                    normalized=norm,
                    link_type=link_type,
                    confidence=confidence,
                    metadata={"aliases": self.alias_table.get(norm, []), "from_stage_a": True},
                )
            )
        for keyword in keyword_candidates(text, limit=10):
            if keyword in seen:
                continue
            seen.add(keyword)
            links.append(
                LinkedEntity(
                    surface=keyword,
                    normalized=keyword,
                    link_type="concept",
                    confidence=0.48,
                    metadata={"aliases": self.alias_table.get(keyword, [])},
                )
            )
        return links
