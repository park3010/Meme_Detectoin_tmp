"""Prompt construction for target, intent, and tactic annotation."""

from __future__ import annotations

import json

from tool.annotation.schemas import UnifiedSample


SYSTEM_PROMPT = """You are a careful research annotator for multimodal meme analysis.
Your task is to annotate target, intent, and tactic using only the meme image and OCR text provided in the user message.
Do not use source metadata, dataset name semantics, platform, timestamp, country, uploader, filename hints, or external hidden assumptions.
Return concise evidence-based labels using the exact controlled vocabulary.
Return valid JSON only. Do not use markdown code fences. Do not reveal chain-of-thought."""


CONTROLLED_VOCABULARY = {
    "multimodal_relation": [
        "text_only",
        "image_only",
        "complementary",
        "incongruent",
        "cross_modal_implication",
    ],
    "target_presence": ["explicit", "implicit", "none"],
    "target_granularity": ["individual", "organization", "community", "society", "none"],
    "protected_attribute": [
        "sex_gender",
        "race_ethnicity",
        "religion",
        "nationality",
        "disability",
        "sexual_orientation",
        "political_ideology",
        "other",
        "none",
    ],
    "intent_primary": [
        "ridicule_mockery",
        "denigration_insult",
        "persuasion_propaganda",
        "mobilization_incitement",
        "misinformation_deception",
        "harassment_threat",
        "solidarity_support",
        "self_expression",
        "entertainment",
        "other",
    ],
    "stance": ["supportive", "neutral", "hostile"],
    "tactic_rhetorical": [
        "stereotype",
        "sarcasm_irony",
        "metaphor",
        "exaggeration",
        "dehumanization",
        "slur",
        "objectification",
        "slogan",
        "fear_appeal",
        "whataboutism",
        "smear",
        "exclusion",
        "conspiracy_cue",
        "dog_whistle",
        "other",
    ],
    "confidence": ["high", "medium", "low"],
    "not_sure": ["yes", "no"],
}


def expected_schema_template(sample: UnifiedSample) -> dict:
    """Return the exact JSON shape expected from the model.

    sample_id and dataset_name are intentionally excluded from the model-facing
    schema; the pipeline injects them after validation.
    """

    return {
        "surface_content": {
            "visual_scene_summary": "",
            "multimodal_relation": "text_only",
        },
        "target": {
            "target_presence": "none",
            "target_granularity": "none",
            "protected_attribute": ["none"],
            "target_text_span": "",
            "target_visual_cue": "",
            "target_explanation_short": "",
        },
        "intent": {
            "intent_primary": "other",
            "stance": "neutral",
            "intent_free_text": "",
            "secondary_intent": "",
            "background_knowledge_needed": False,
            "background_knowledge_text": "",
        },
        "tactic": {
            "tactic_rhetorical": ["other"],
            "tactic_multimodal_relation": "text_only",
            "evidence_text_span": "",
            "evidence_image_region_description": "",
            "tactic_explanation_short": "",
        },
        "evidence": {
            "key_text_evidence": "",
            "key_visual_evidence": "",
            "key_cross_modal_evidence": "",
        },
        "quality": {
            "confidence": "low",
            "not_sure": "yes",
        },
    }


def build_user_prompt(sample: UnifiedSample, multimodal: bool = True) -> str:
    """Build the user prompt for a sample."""

    mode_note = "The image is attached." if multimodal else "No image is attached; use OCR text only."
    schema_json = json.dumps(expected_schema_template(sample), ensure_ascii=False, indent=2)
    vocab_json = json.dumps(CONTROLLED_VOCABULARY, ensure_ascii=False, indent=2)
    return f"""Annotate this meme for target, intent, and tactic.

Available inputs:
- OCR text: {sample.ocr_text or "[EMPTY OCR]"}
- Image availability: {mode_note}

Important evidence boundary:
- You are not given a meaningful dataset name, label, platform, timestamp, country, uploader, filename, or source metadata.
- Do not infer anything from hidden dataset identity, sample IDs, raw labels, filenames, or benchmark conventions.
- Use only the OCR text above and, when attached, the image pixels.
- Do not copy the full OCR text into your JSON. The pipeline will attach ocr_text_full after validation.
- In text-only mode, leave visual_scene_summary as an empty string rather than guessing image content.

Definitions:
- target = who or what is being targeted by the meme.
- intent = what the meme is trying to do communicatively.
- tactic = rhetorical, persuasive, or attack device used to produce the effect.
- explicit target = directly named or clearly depicted.
- implicit target = inferable from text/image relation without being directly named.
- none = no identifiable target.
- multimodal relation:
  - text_only: annotation evidence comes only from OCR text.
  - image_only: annotation evidence comes only from image content.
  - complementary: text and image reinforce the same meaning.
  - incongruent: text and image conflict or create irony through mismatch.
  - cross_modal_implication: the key meaning emerges only by combining text and image.

Rules:
1. Use only the meme image and OCR text. Do not use metadata, labels, dataset-name semantics, filenames, or source assumptions.
2. Use the exact labels in the controlled vocabulary.
3. If uncertain, use lower confidence and set not_sure to "yes".
4. If no target is present, set target_presence to "none", target_granularity to "none", and protected_attribute to ["none"].
5. Keep explanations short and evidence-based.
6. Return valid JSON only, with no markdown and no chain-of-thought.
7. Do not include sample_id, dataset_name, raw_label, image filename, or any metadata fields in your JSON.
8. Do not include ocr_text_full in your JSON.

Controlled vocabulary:
{vocab_json}

Return exactly this JSON shape, replacing defaults with your annotation:
{schema_json}"""


def build_repair_prompt(raw_response: str, validation_error: str, sample: UnifiedSample) -> str:
    """Build a prompt asking the model to repair invalid JSON."""

    schema_json = json.dumps(expected_schema_template(sample), ensure_ascii=False, indent=2)
    return f"""Repair or regenerate the annotation as valid JSON only.
Use only the OCR text below and the required schema. Preserve intended labels from the invalid output only when recoverable.
Do not copy the full OCR text into your JSON. Do not include ocr_text_full.
Do not add markdown code fences or explanations.

OCR text:
{sample.ocr_text or "[EMPTY OCR]"}

Validation error:
{validation_error}

Required schema:
{schema_json}

Invalid output:
{raw_response}"""
