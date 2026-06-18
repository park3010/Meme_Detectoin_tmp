"""OpenAI Responses API annotator client for meme annotation."""

from __future__ import annotations

import base64
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from tool.annotation.config import (
    MAX_OUTPUT_TOKENS,
    MAX_RETRIES,
    MODEL_NAME,
    OPENAI_API_KEY,
    RETRY_BACKOFF_SECONDS,
    TIMEOUT_SECONDS,
)
from tool.annotation.prompt_builder import build_repair_prompt
from tool.annotation.schemas import AnnotatorResult, MemeAnnotation, UnifiedSample
from tool.annotation.utils import setup_logger


logger = setup_logger(__name__)

DEFAULT_GPT55_MODEL = "gpt-5.5"
DEFAULT_REASONING_EFFORT = "low"

ANNOTATION_RESPONSE_FORMAT: Dict[str, Any] = {
    "format": {
        "type": "json_schema",
        "name": "meme_target_intent_tactic_annotation",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["surface_content", "target", "intent", "tactic", "evidence", "quality"],
            "properties": {
                "surface_content": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["visual_scene_summary", "multimodal_relation"],
                    "properties": {
                        "visual_scene_summary": {"type": "string"},
                        "multimodal_relation": {
                            "type": "string",
                            "enum": [
                                "text_only",
                                "image_only",
                                "complementary",
                                "incongruent",
                                "cross_modal_implication",
                            ],
                        },
                    },
                },
                "target": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "target_presence",
                        "target_granularity",
                        "protected_attribute",
                        "target_text_span",
                        "target_visual_cue",
                        "target_explanation_short",
                    ],
                    "properties": {
                        "target_presence": {"type": "string", "enum": ["explicit", "implicit", "none"]},
                        "target_granularity": {
                            "type": "string",
                            "enum": ["individual", "organization", "community", "society", "none"],
                        },
                        "protected_attribute": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
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
                            },
                        },
                        "target_text_span": {"type": "string"},
                        "target_visual_cue": {"type": "string"},
                        "target_explanation_short": {"type": "string"},
                    },
                },
                "intent": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "intent_primary",
                        "stance",
                        "intent_free_text",
                        "secondary_intent",
                        "background_knowledge_needed",
                        "background_knowledge_text",
                    ],
                    "properties": {
                        "intent_primary": {
                            "type": "string",
                            "enum": [
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
                        },
                        "stance": {"type": "string", "enum": ["supportive", "neutral", "hostile"]},
                        "intent_free_text": {"type": "string"},
                        "secondary_intent": {"type": "string"},
                        "background_knowledge_needed": {"type": "boolean"},
                        "background_knowledge_text": {"type": "string"},
                    },
                },
                "tactic": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "tactic_rhetorical",
                        "tactic_multimodal_relation",
                        "evidence_text_span",
                        "evidence_image_region_description",
                        "tactic_explanation_short",
                    ],
                    "properties": {
                        "tactic_rhetorical": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
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
                            },
                        },
                        "tactic_multimodal_relation": {
                            "type": "string",
                            "enum": [
                                "text_only",
                                "image_only",
                                "complementary",
                                "incongruent",
                                "cross_modal_implication",
                            ],
                        },
                        "evidence_text_span": {"type": "string"},
                        "evidence_image_region_description": {"type": "string"},
                        "tactic_explanation_short": {"type": "string"},
                    },
                },
                "evidence": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["key_text_evidence", "key_visual_evidence", "key_cross_modal_evidence"],
                    "properties": {
                        "key_text_evidence": {"type": "string"},
                        "key_visual_evidence": {"type": "string"},
                        "key_cross_modal_evidence": {"type": "string"},
                    },
                },
                "quality": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["confidence", "not_sure"],
                    "properties": {
                        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                        "not_sure": {"type": "string", "enum": ["yes", "no"]},
                    },
                },
            },
        },
    }
}


class BaseAnnotatorClient(ABC):
    """Abstract annotator interface."""

    @abstractmethod
    def annotate(self, sample: UnifiedSample, system_prompt: str, user_prompt: str) -> AnnotatorResult:
        """Annotate one sample and return raw plus validated output."""


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Parse a JSON object from raw model text."""

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("Model output is not a JSON object")
    return obj


def _validate_annotation(raw_text: str, sample: UnifiedSample) -> MemeAnnotation:
    """Validate model JSON and attach storage-only identifiers."""

    obj = _extract_json_object(raw_text)
    obj["sample_id"] = sample.sample_id
    obj["dataset_name"] = sample.dataset_name
    surface_content = obj.setdefault("surface_content", {})
    if isinstance(surface_content, dict):
        surface_content["ocr_text_full"] = sample.ocr_text
    return MemeAnnotation.parse_obj(obj)


def _image_to_data_url(path: str) -> str:
    """Encode an image as a data URL for the Responses API."""

    image_path = Path(path)
    suffix = image_path.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix == "jpg" else suffix
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return "data:image/{mime};base64,{data}".format(mime=mime, data=data)


def _response_text(response: Any) -> str:
    """Extract text from an OpenAI Responses API response object."""

    if isinstance(response, dict):
        output_text = response.get("output_text")
        if output_text:
            return str(output_text)
        chunks: List[str] = []
        for item in response.get("output", []) or []:
            for content in item.get("content", []) or []:
                text = content.get("text")
                if text:
                    chunks.append(str(text))
        return "\n".join(chunks).strip()

    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)

    chunks: List[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(str(text))
    return "\n".join(chunks).strip()


class OpenAIAnnotatorClient(BaseAnnotatorClient):
    """GPT-5.5 annotator implemented with the OpenAI Responses API."""

    def __init__(
        self,
        api_key: str = OPENAI_API_KEY,
        model: str = MODEL_NAME,
        multimodal: bool = True,
        max_retries: int = MAX_RETRIES,
        timeout: float = TIMEOUT_SECONDS,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install the OpenAI SDK with: pip install openai") from exc

        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.model = model or DEFAULT_GPT55_MODEL
        self.multimodal = multimodal
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort

    def annotate(self, sample: UnifiedSample, system_prompt: str, user_prompt: str) -> AnnotatorResult:
        """Annotate one sample, then attempt one JSON repair call if needed."""

        raw_response = ""
        try:
            raw_response = self._call_with_retries(sample, system_prompt, user_prompt, include_image=self.multimodal)
            annotation = _validate_annotation(raw_response, sample)
            if not self.multimodal:
                annotation.surface_content.visual_scene_summary = ""
            return AnnotatorResult(sample_id=sample.sample_id, raw_response=raw_response, annotation=annotation)
        except Exception as first_error:
            logger.warning("Initial annotation failed for %s: %s", sample.sample_id, first_error)
            try:
                repair_prompt = build_repair_prompt(raw_response, str(first_error), sample)
                repaired = self._call_with_retries(sample, system_prompt, repair_prompt, include_image=False)
                annotation = _validate_annotation(repaired, sample)
                if not self.multimodal:
                    annotation.surface_content.visual_scene_summary = ""
                return AnnotatorResult(sample_id=sample.sample_id, raw_response=repaired, annotation=annotation)
            except Exception as repair_error:
                return AnnotatorResult(
                    sample_id=sample.sample_id,
                    raw_response=raw_response,
                    annotation=None,
                    error=str(repair_error),
                    error_type=type(repair_error).__name__,
                )

    def _call_with_retries(
        self,
        sample: UnifiedSample,
        system_prompt: str,
        user_prompt: str,
        include_image: bool,
    ) -> str:
        """Call the API with simple retry backoff."""

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._call_once(sample, system_prompt, user_prompt, include_image=include_image)
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                sleep_for = RETRY_BACKOFF_SECONDS * attempt
                logger.warning(
                    "Responses API call failed for %s on attempt %s/%s: %s; retrying in %.1fs",
                    sample.sample_id,
                    attempt,
                    self.max_retries,
                    exc,
                    sleep_for,
                )
                time.sleep(sleep_for)
        assert last_error is not None
        raise last_error

    def _call_once(self, sample: UnifiedSample, system_prompt: str, user_prompt: str, include_image: bool) -> str:
        """Send one request through the OpenAI Responses API."""

        input_content: List[Dict[str, Any]] = [{"type": "input_text", "text": user_prompt}]

        if include_image and sample.image_path and Path(sample.image_path).exists():
            input_content.append(
                {
                    "type": "input_image",
                    "image_url": _image_to_data_url(sample.image_path),
                    "detail": "low",
                }
            )

        response = self.client.responses.create(
            model=self.model,
            reasoning={"effort": self.reasoning_effort},
            max_output_tokens=MAX_OUTPUT_TOKENS,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user", "content": input_content},
            ],
            text=ANNOTATION_RESPONSE_FORMAT,
        )

        message = _response_text(response)
        if not message:
            raise ValueError("Empty model response")
        return message
