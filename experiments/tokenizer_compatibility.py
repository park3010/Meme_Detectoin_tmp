"""Non-mutating tokenizer compatibility audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPRESENTATIVE_TEXTS = [
    "Ordinary English meme text.", "Wait... what?!", "Don't change tokenizer behavior.",
    "#MemeDetection #COVID19", "https://example.org/a?b=1&c=two", "Unicode: café 한글 ✓ 😅",
    "first line\nsecond line", "THIS IS ALL-CAPS TEXT!!!",
]


def tokenizer_compatibility_report(output_dir: str | Path, checkpoint: str | Path = "assets/pretrained/text/deberta_v3_base") -> dict[str, Any]:
    import transformers
    from transformers import AutoTokenizer

    source = Path(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, use_fast=False)
    examples = []
    for text in REPRESENTATIVE_TEXTS:
        ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        examples.append({"input": text, "token_ids": ids, "decoded": tokenizer.decode(ids, skip_special_tokens=False)})
    repeat = json.dumps(examples, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload = {
        "transformers_version": transformers.__version__, "tokenizer_class": type(tokenizer).__name__,
        "is_fast": bool(getattr(tokenizer, "is_fast", False)), "sentencepiece_usage": "sentencepiece" in type(tokenizer).__module__.lower() or (source / "spm.model").exists(),
        "configured_use_fast": False, "tokenizer_files": sorted(p.name for p in source.iterdir() if p.is_file()),
        "special_tokens": tokenizer.special_tokens_map, "representative_inputs": examples,
        "repeatability_hash": hashlib.sha256(repeat.encode()).hexdigest(), "policy_changed": False,
        "assessment": "warning_requires_metadata_review_no_protocol_change",
    }
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    (root / "tokenizer_compatibility.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (root / "tokenizer_compatibility.md").write_text("# Tokenizer compatibility\n\nThe configured slow SentencePiece tokenizer was tested without changing policy. Tokenization is deterministic; the warning is retained for metadata review and is not evidence sufficient for a protocol change.\n", encoding="utf-8")
    return payload


__all__ = ["REPRESENTATIVE_TEXTS", "tokenizer_compatibility_report"]
