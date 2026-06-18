"""Pydantic schemas for samples and model annotations."""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, validator


MultimodalRelation = Literal[
    "text_only",
    "image_only",
    "complementary",
    "incongruent",
    "cross_modal_implication",
]
TargetPresence = Literal["explicit", "implicit", "none"]
TargetGranularity = Literal["individual", "organization", "community", "society", "none"]
ProtectedAttribute = Literal[
    "sex_gender",
    "race_ethnicity",
    "religion",
    "nationality",
    "disability",
    "sexual_orientation",
    "political_ideology",
    "other",
    "none",
]
IntentPrimary = Literal[
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
]
Stance = Literal["supportive", "neutral", "hostile"]
TacticLabel = Literal[
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
]
Confidence = Literal["high", "medium", "low"]
YesNo = Literal["yes", "no"]


class UnifiedSample(BaseModel):
    """A dataset-agnostic meme sample."""

    sample_id: str
    dataset_name: str
    image_path: Optional[str] = None
    ocr_text: str = ""
    raw_label: Optional[Union[int, str]] = None
    raw_record: dict[str, Any] = Field(default_factory=dict)


class SurfaceContent(BaseModel):
    ocr_text_full: str = ""
    visual_scene_summary: str = ""
    multimodal_relation: MultimodalRelation = "text_only"


class TargetAnnotation(BaseModel):
    target_presence: TargetPresence = "none"
    target_granularity: TargetGranularity = "none"
    protected_attribute: list[ProtectedAttribute] = Field(default_factory=lambda: ["none"])
    target_text_span: str = ""
    target_visual_cue: str = ""
    target_explanation_short: str = ""

    @validator("protected_attribute", pre=True, always=True)
    def ensure_non_empty_attributes(cls, value: object) -> list[str]:
        if value is None or value == "":
            return ["none"]
        if isinstance(value, str):
            return [value]
        if isinstance(value, list) and value:
            return value
        return ["none"]


class IntentAnnotation(BaseModel):
    intent_primary: IntentPrimary = "other"
    stance: Stance = "neutral"
    intent_free_text: str = ""
    secondary_intent: str = ""
    background_knowledge_needed: bool = False
    background_knowledge_text: str = ""


class TacticAnnotation(BaseModel):
    tactic_rhetorical: list[TacticLabel] = Field(default_factory=lambda: ["other"])
    tactic_multimodal_relation: MultimodalRelation = "text_only"
    evidence_text_span: str = ""
    evidence_image_region_description: str = ""
    tactic_explanation_short: str = ""

    @validator("tactic_rhetorical", pre=True, always=True)
    def ensure_non_empty_tactics(cls, value: object) -> list[str]:
        if value is None or value == "":
            return ["other"]
        if isinstance(value, str):
            return [value]
        if isinstance(value, list) and value:
            return value
        return ["other"]


class EvidenceAnnotation(BaseModel):
    key_text_evidence: str = ""
    key_visual_evidence: str = ""
    key_cross_modal_evidence: str = ""


class QualityAnnotation(BaseModel):
    confidence: Confidence = "low"
    not_sure: YesNo = "yes"


class MemeAnnotation(BaseModel):
    sample_id: str
    dataset_name: str
    surface_content: SurfaceContent = Field(default_factory=SurfaceContent)
    target: TargetAnnotation = Field(default_factory=TargetAnnotation)
    intent: IntentAnnotation = Field(default_factory=IntentAnnotation)
    tactic: TacticAnnotation = Field(default_factory=TacticAnnotation)
    evidence: EvidenceAnnotation = Field(default_factory=EvidenceAnnotation)
    quality: QualityAnnotation = Field(default_factory=QualityAnnotation)


class AnnotatorResult(BaseModel):
    sample_id: str
    raw_response: str = ""
    annotation: Optional[MemeAnnotation] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
