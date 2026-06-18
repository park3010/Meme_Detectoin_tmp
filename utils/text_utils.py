"""Small text processing helpers with no heavyweight dependencies."""

from __future__ import annotations

import re
from collections import Counter


TOKEN_RE = re.compile(r"[A-Za-z0-9_#@']+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "them",
    "this",
    "to",
    "with",
    "you",
    "your",
}

RHETORICAL_CUE_PATTERNS = {
    "sarcasm_irony": [
        r"\b(sure|totally|yeah right|as if|then i told|dankest)\b",
        r"\?",
    ],
    "exaggeration": [
        r"\b(always|never|everyone|everything|nothing|highly rated|train wreck)\b",
        r"!{2,}",
    ],
    "smear": [
        r"\b(shit[- ]?show|trash|corrupt|traitor|idiot|stupid)\b",
    ],
    "fear_appeal": [
        r"\b(threat|danger|destroy|invasion|virus|contagion)\b",
    ],
    "stereotype": [
        r"\b(real men|cultural difference|all women|all men|these people)\b",
    ],
    "dehumanization": [
        r"\b(animals?|vermin|parasites?|cockroach)\b",
    ],
    "slur": [
        r"\b(slur|retard|fag)\b",
    ],
}

TARGET_CUE_TERMS = {
    "people",
    "women",
    "men",
    "muslim",
    "jew",
    "christian",
    "black",
    "white",
    "asian",
    "immigrant",
    "thais",
    "european",
    "america",
    "china",
    "government",
    "schools",
    "media",
    "party",
}


def normalize_text(text: object) -> str:
    """Collapse whitespace and coerce arbitrary values to a clean string."""

    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def tokenize(text: str, lowercase: bool = True) -> list[str]:
    """Tokenize text into a simple alphanumeric stream."""

    tokens = TOKEN_RE.findall(normalize_text(text))
    return [token.lower() for token in tokens] if lowercase else tokens


def keyword_candidates(text: str, limit: int = 12) -> list[str]:
    """Return high-signal lexical candidates for queries or linking."""

    tokens = [tok for tok in tokenize(text) if tok not in STOPWORDS and len(tok) > 2]
    counts = Counter(tokens)
    return [token for token, _ in counts.most_common(limit)]


def sentence_chunks(text: str, limit: int = 8, min_chars: int = 8) -> list[str]:
    """Split OCR into compact sentence-like chunks for evidence tokens."""

    clean = normalize_text(text)
    if not clean:
        return []
    chunks = re.split(r"(?<=[.!?])\s+|\n+| {2,}", clean)
    normalized = [normalize_text(chunk) for chunk in chunks if len(normalize_text(chunk)) >= min_chars]
    if normalized:
        return normalized[:limit]
    words = clean.split()
    return [" ".join(words[idx : idx + 12]) for idx in range(0, min(len(words), limit * 12), 12)][:limit]


def capitalized_spans(text: str, limit: int = 8) -> list[str]:
    """Extract rough entity-like spans based on capitalization."""

    clean = normalize_text(text)
    spans = re.findall(r"(?:[A-Z][A-Za-z0-9_#@']+\s*){1,4}", clean)
    seen: set[str] = set()
    results: list[str] = []
    for span in spans:
        value = normalize_text(span)
        if value and value.lower() not in STOPWORDS and value not in seen:
            seen.add(value)
            results.append(value)
        if len(results) >= limit:
            break
    return results


def rhetorical_cues(text: str) -> dict[str, float]:
    """Detect rough rhetorical cue strengths in OCR text."""

    lowered = normalize_text(text).lower()
    cues: dict[str, float] = {}
    for label, patterns in RHETORICAL_CUE_PATTERNS.items():
        matches = sum(1 for pattern in patterns if re.search(pattern, lowered, flags=re.IGNORECASE))
        if matches:
            cues[label] = min(1.0, 0.45 + 0.25 * matches)
    return cues


def target_presence_score(text: str) -> float:
    """Estimate whether OCR names or implies a target."""

    lowered_tokens = set(tokenize(text))
    entity_bonus = 0.35 if capitalized_spans(text, limit=2) else 0.0
    lex_bonus = 0.45 if lowered_tokens & TARGET_CUE_TERMS else 0.0
    pronoun_bonus = 0.15 if lowered_tokens & {"they", "them", "those", "these"} else 0.0
    return min(1.0, entity_bonus + lex_bonus + pronoun_bonus)


def language_hint(text: str) -> str:
    """Return a simple language/script hint for compatibility scoring."""

    clean = normalize_text(text)
    if not clean:
        return "unknown"
    ascii_ratio = sum(1 for ch in clean if ord(ch) < 128) / max(1, len(clean))
    if ascii_ratio > 0.92:
        return "latin"
    if re.search(r"[\uac00-\ud7af]", clean):
        return "korean"
    if re.search(r"[\u4e00-\u9fff]", clean):
        return "cjk"
    if re.search(r"[\u0600-\u06ff]", clean):
        return "arabic"
    return "mixed"


def jaccard_similarity(a: str, b: str) -> float:
    """Compute lexical Jaccard similarity for lightweight scoring."""

    left = set(tokenize(a))
    right = set(tokenize(b))
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))
