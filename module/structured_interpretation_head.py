"""Stage E: structured interpretation head."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from module.evidence_fusion_reasoning import StageDOutput
from module.internal_evidence_extractor import StageAOutput
from module.knowledge_filter_verifier import StageCOutput
from utils.text_utils import capitalized_spans, keyword_candidates, rhetorical_cues, target_presence_score


# =============================================================================
# Stage E schemas
# =============================================================================

@dataclass
class Prediction:
    """A labeled prediction with score distribution."""

    label: str
    score: float
    scores: dict[str, float] = field(default_factory=dict)
    logits: torch.Tensor | None = None
    labels: list[str] = field(default_factory=list)
    auxiliary: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageEMetadata:
    """Auditable structured-output provenance metadata."""

    internal_evidence_count: int
    external_evidence_count: int
    rationale_backend: str = "template"
    prediction_fields: list[str] = field(default_factory=list)
    field_provenance: dict[str, str] = field(default_factory=dict)
    label_spaces: dict[str, list[str]] = field(default_factory=dict)
    trainable_logits_fields: list[str] = field(default_factory=list)
    proxy_fields: list[str] = field(default_factory=list)
    template_fields: list[str] = field(default_factory=list)
    cue_fields: list[str] = field(default_factory=list)
    stage_d_trace_available: bool = False
    evidence_attribution_backend: str = "gate_attention_score_proxy"
    output_contract_version: str = "stage_e_structured_output_v1"


@dataclass
class StageEOutput:
    """Final structured interpretation output."""

    sample_id: str
    dataset_name: str
    harmfulness: Prediction
    target: Prediction
    intent: Prediction
    tactic: Prediction
    supporting_evidence: dict[str, list[dict[str, Any]]]
    rationale: str
    structured_prediction: dict[str, Any]
    metadata: StageEMetadata


# =============================================================================
# Prediction helpers
# =============================================================================



# =============================================================================
# Harmfulness / target / intent / tactic heads
# =============================================================================

class HarmfulnessHead(nn.Module):
    """Binary harmfulness head with neural logits and interpretable lexical priors."""

    labels = ["non_harmful", "harmful"]

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, len(self.labels))

    def compute_logits(self, shared_state: torch.Tensor, text: str = "") -> torch.Tensor:
        """Return trainable logits with lightweight lexical priors added."""

        logits = self.classifier(shared_state.float())
        lowered = text.lower()
        harmful_cues = ["hate", "kill", "stupid", "inferior", "threat", "shit", "trash", "blame", "low on"]
        supportive_cues = ["support", "applause", "solidarity", "thanks"]
        if any(cue in lowered for cue in harmful_cues):
            logits[1] += 0.9
        if any(cue in lowered for cue in supportive_cues):
            logits[0] += 0.5
        return logits

    def forward(self, shared_state: torch.Tensor, text: str = "") -> dict[str, float]:
        logits = self.compute_logits(shared_state, text)
        probs = torch.softmax(logits, dim=-1).detach()
        return {label: float(probs[idx]) for idx, label in enumerate(self.labels)}

class TargetHead(nn.Module):
    """Predict target granularity/presence."""

    labels = ["none", "individual", "organization", "community", "society"]
    presence_labels = ["explicit", "implicit", "none"]

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, len(self.labels))
        self.presence_classifier = nn.Linear(hidden_dim, len(self.presence_labels))

    def compute_logits(self, target_latent: torch.Tensor, text: str = "") -> torch.Tensor:
        """Return trainable target granularity logits with fallback priors."""

        logits = self.classifier(target_latent.float())
        lowered = text.lower()
        if capitalized_spans(text, limit=2):
            logits[self.labels.index("individual")] += 0.35
        if any(term in lowered for term in ["people", "women", "men", "thais", "european", "america"]):
            logits[self.labels.index("community")] += 0.45
        if any(term in lowered for term in ["government", "school", "party", "media"]):
            logits[self.labels.index("organization")] += 0.35
        return logits

    def compute_presence_logits(self, target_latent: torch.Tensor, text: str = "") -> torch.Tensor:
        """Return trainable target-presence logits with small lexical priors."""

        logits = self.presence_classifier(target_latent.float())
        presence_score = target_presence_score(text)
        entities = capitalized_spans(text, limit=2)
        if entities:
            logits[self.presence_labels.index("explicit")] += 0.25
        if not entities and 0.25 <= presence_score < 0.55:
            logits[self.presence_labels.index("implicit")] += 0.2
        if not entities and presence_score < 0.25:
            logits[self.presence_labels.index("none")] += 0.2
        return logits

    def forward(self, target_latent: torch.Tensor, text: str = "") -> dict[str, float]:
        logits = self.compute_logits(target_latent, text)
        probs = torch.softmax(logits, dim=-1).detach()
        return {label: float(probs[idx]) for idx, label in enumerate(self.labels)}

class IntentHead(nn.Module):
    """Predict primary communicative intent."""

    labels = [
        "ridicule_mockery",
        "denigration_insult",
        "harassment_threat",
        "persuasion_propaganda",
        "misinformation_deception",
        "mobilization_incitement",
        "solidarity_support",
        "self_expression",
        "entertainment",
        "criticism",
        "neutral",
        "other",
    ]

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, len(self.labels))

    def compute_logits(self, intent_latent: torch.Tensor, text: str = "") -> torch.Tensor:
        """Return trainable intent logits with fallback lexical priors."""

        logits = self.classifier(intent_latent.float())
        lowered = text.lower()
        if any(term in lowered for term in ["shit", "stupid", "laugh", "lol", "joke"]):
            _add_prior(logits, self.labels, "ridicule_mockery", 0.6)
        if any(term in lowered for term in ["vote", "propaganda", "change", "america"]):
            _add_prior(logits, self.labels, "persuasion_propaganda", 0.25)
        if any(term in lowered for term in ["support", "applause", "thanks"]):
            _add_prior(logits, self.labels, "solidarity_support", 0.6)
        if any(term in lowered for term in ["join", "march", "fight back", "rise", "mobilize"]):
            _add_prior(logits, self.labels, "mobilization_incitement", 0.35)
        if any(term in lowered for term in ["criticize", "critique", "wrong", "failed", "corrupt"]):
            _add_prior(logits, self.labels, "criticism", 0.3)
        return logits

    def forward(self, intent_latent: torch.Tensor, text: str = "") -> dict[str, float]:
        logits = self.compute_logits(intent_latent, text)
        probs = torch.softmax(logits, dim=-1).detach()
        return {label: float(probs[idx]) for idx, label in enumerate(self.labels)}


def _add_prior(logits: torch.Tensor, labels: list[str], label: str, amount: float) -> None:
    """Safely add a lexical prior when the label exists."""

    if label in labels:
        logits[labels.index(label)] += amount

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


# =============================================================================
# Evidence attribution
# =============================================================================

class EvidenceAttributionLayer:
    """Select supporting internal and external evidence items."""

    def select(
        self,
        stage_a: StageAOutput,
        stage_c: StageCOutput,
        stage_d: StageDOutput,
        top_k: int = 3,
        task: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Return selected evidence dictionaries."""

        token_gates = stage_d.gates.get("token_level")
        if isinstance(token_gates, torch.Tensor):
            token_gates = token_gates.detach()
        else:
            token_gates = torch.ones(len(stage_a.evidence_items))
        head_level = stage_d.gates.get("head_level")
        task_idx = {"target": 0, "intent": 1, "tactic": 2}.get(task or "", None)
        internal_ranked = []
        for item in stage_a.evidence_items:
            gate_score = float(token_gates[item.token_index]) if item.token_index < token_gates.numel() else item.score
            if isinstance(head_level, torch.Tensor) and task_idx is not None and item.token_index < head_level.size(1):
                gate_score = 0.7 * gate_score + 0.3 * float(head_level.detach()[task_idx, item.token_index])
            type_bonus = 1.1 if item.evidence_type in {"text_span", "cross_modal_incongruity", "local_symbol"} else 1.0
            internal_ranked.append((gate_score * max(item.score, 0.05) * type_bonus, item))
        internal_ranked.sort(key=lambda pair: pair[0], reverse=True)
        internal = [
            {
                "evidence_id": item.evidence_id,
                "type": item.evidence_type,
                "text": item.text,
                "score": float(score),
                "modality": item.metadata.get("modality"),
                "grounding_type": item.metadata.get("grounding_type"),
                "is_heuristic": item.metadata.get("is_heuristic"),
                "source_stage": item.metadata.get("source_stage", "stage_a"),
                "attribution_backend": "gate_attention_score_proxy",
                "metadata": item.metadata,
            }
            for score, item in internal_ranked[:top_k]
        ]

        attention = stage_d.cross_attention_weights.detach() if isinstance(stage_d.cross_attention_weights, torch.Tensor) else torch.zeros(0, 0)
        external_ranked = []
        for idx, item in enumerate(stage_c.verified_items):
            attention_score = float(attention[:, idx].mean()) if attention.numel() and idx < attention.size(1) else 0.0
            support_bonus = 0.08 if item.support_label == "support" else -0.02 if item.support_label == "contradict" else 0.0
            external_ranked.append((item.final_score + 0.25 * attention_score + support_bonus, attention_score, item))
        external_ranked.sort(key=lambda row: row[0], reverse=True)
        external = [
            {
                "knowledge_id": item.knowledge_id,
                "text": item.text,
                "source": item.source,
                "score": float(score),
                "final_score": item.final_score,
                "attention_score": float(attention_score),
                "support_label": item.support_label,
                "candidate_origin": item.metadata.get("candidate_origin"),
                "is_external_knowledge": item.metadata.get("is_external_knowledge"),
                "is_generated": item.metadata.get("is_generated"),
                "is_fallback": item.metadata.get("is_fallback"),
                "is_retrieved": item.metadata.get("is_retrieved"),
                "verification_status": item.metadata.get("verification_status"),
                "attribution_backend": "final_score_attention_support_proxy",
                "metadata": item.metadata,
            }
            for score, attention_score, item in external_ranked[:top_k]
        ]
        return {"internal": internal, "external": external}


# =============================================================================
# Deterministic rationale rendering
# =============================================================================

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


# =============================================================================
# Structured output provenance helpers
# =============================================================================



# =============================================================================
# StructuredInterpretationHead
# =============================================================================

class StructuredInterpretationHead(nn.Module):
    """Predict structured harmful meme interpretation and rationale."""

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.harmfulness = HarmfulnessHead(hidden_dim)
        self.target = TargetHead(hidden_dim)
        self.intent = IntentHead(hidden_dim)
        self.tactic = TacticHead(hidden_dim)
        self.attribution = EvidenceAttributionLayer()
        self.rationale = TemplateRationaleGenerator()

    def forward(self, stage_a: StageAOutput, stage_c: StageCOutput, stage_d: StageDOutput) -> StageEOutput:
        """Run final structured interpretation."""

        ocr_text = _ocr_text(stage_a)
        harm_pred = _to_prediction_from_logits(self.harmfulness.compute_logits(stage_d.shared_reasoning_state, ocr_text), self.harmfulness.labels)
        target_pred = _to_prediction_from_logits(self.target.compute_logits(stage_d.task_latents["target"], ocr_text), self.target.labels)
        target_presence_pred = _to_prediction_from_logits(
            self.target.compute_presence_logits(stage_d.task_latents["target"], ocr_text),
            self.target.presence_labels,
        )
        intent_pred = _to_prediction_from_logits(self.intent.compute_logits(stage_d.task_latents["intent"], ocr_text), self.intent.labels)
        tactic_pred = _to_prediction_from_logits(self.tactic.compute_logits(stage_d.task_latents["tactic"], ocr_text), self.tactic.labels)
        stage_a_relation = _stage_a_multimodal_relation(stage_a)
        tactic_relation_pred = _to_prediction_from_logits(
            self.tactic.compute_relation_logits(
                stage_d.task_latents["tactic"],
                ocr_text,
                stage_a_relation=stage_a_relation,
            ),
            self.tactic.relation_labels,
        )
        evidence = self.attribution.select(stage_a, stage_c, stage_d)
        rationale = self.rationale.generate(harm_pred, target_pred, intent_pred, tactic_pred, evidence)
        harmfulness_payload = _harmfulness_payload(harm_pred)
        target_payload = _target_payload(target_pred, stage_a, target_presence_pred)
        intent_payload = _intent_payload(intent_pred, harm_pred, stage_a, stage_c)
        tactic_payload = _tactic_payload(tactic_pred, stage_a, tactic_relation_pred)
        label_spaces = _label_spaces(
            harm_pred,
            target_pred,
            intent_pred,
            tactic_pred,
            target_presence_pred,
            tactic_relation_pred,
        )
        stage_d_trace_available = bool(getattr(stage_d.metadata, "attention_trace", None))
        regularizer_hook_mode = getattr(stage_d.metadata, "regularizer_hook_mode", "detached_analysis_only")
        output_provenance = {
            "field_provenance": _field_provenance(),
            "trainable_logits_fields": _trainable_logits_fields(),
            "proxy_fields": _proxy_fields(),
            "template_fields": _template_fields(),
            "cue_fields": _cue_fields(),
            "label_spaces": label_spaces,
            "evidence_attribution_backend": "gate_attention_score_proxy",
            "rationale_backend": "template",
            "stage_d_trace_available": stage_d_trace_available,
            "stage_d_regularizer_hook_mode": regularizer_hook_mode,
        }
        structured = {
            "sample_id": stage_a.sample_id,
            "dataset_name": stage_a.dataset_name,
            "harmfulness": harmfulness_payload,
            "target": target_payload,
            "intent": intent_payload,
            "tactic": tactic_payload,
            "supporting_evidence": evidence,
            "rationale": rationale,
            "output_provenance": output_provenance,
            "training_hooks": {
                "harmfulness_scores": harm_pred.scores,
                "target_scores": target_pred.scores,
                "target_presence_scores": target_presence_pred.scores,
                "intent_scores": intent_pred.scores,
                "tactic_scores": tactic_pred.scores,
                "tactic_multimodal_relation_scores": tactic_relation_pred.scores,
                "harmfulness_logits": harm_pred.logits,
                "target_logits": target_pred.logits,
                "target_presence_logits": target_presence_pred.logits,
                "intent_logits": intent_pred.logits,
                "tactic_logits": tactic_pred.logits,
                "tactic_multimodal_relation_logits": tactic_relation_pred.logits,
                "label_spaces": label_spaces,
                "stage_d_regularizers": stage_d.metadata.regularizer_hooks,
                "field_provenance": _field_provenance(),
                "trainable_logits_fields": _trainable_logits_fields(),
                "proxy_fields": _proxy_fields(),
                "stage_d_trace_available": stage_d_trace_available,
                "stage_d_regularizer_hook_mode": regularizer_hook_mode,
            },
        }
        return StageEOutput(
            sample_id=stage_a.sample_id,
            dataset_name=stage_a.dataset_name,
            harmfulness=harm_pred,
            target=target_pred,
            intent=intent_pred,
            tactic=tactic_pred,
            supporting_evidence=evidence,
            rationale=rationale,
            structured_prediction=structured,
            metadata=StageEMetadata(
                internal_evidence_count=len(stage_a.evidence_items),
                external_evidence_count=len(stage_c.verified_items),
                prediction_fields=list(_field_provenance()),
                field_provenance=_field_provenance(),
                label_spaces=label_spaces,
                trainable_logits_fields=_trainable_logits_fields(),
                proxy_fields=_proxy_fields(),
                template_fields=_template_fields(),
                cue_fields=_cue_fields(),
                stage_d_trace_available=stage_d_trace_available,
                evidence_attribution_backend="gate_attention_score_proxy",
                rationale_backend="template",
                output_contract_version="stage_e_structured_output_v1",
            ),
        )


def _to_prediction(scores: dict[str, float]) -> Prediction:
    label, score = max(scores.items(), key=lambda item: item[1])
    return Prediction(label=label, score=float(score), scores=scores, labels=list(scores))


def _to_prediction_from_logits(logits: torch.Tensor, labels: list[str]) -> Prediction:
    probs = torch.softmax(logits, dim=-1)
    idx = int(torch.argmax(probs.detach()).item())
    scores = {label: float(probs.detach()[i]) for i, label in enumerate(labels)}
    return Prediction(
        label=labels[idx],
        score=float(probs.detach()[idx]),
        scores=scores,
        logits=logits,
        labels=list(labels),
    )


def _ocr_text(stage_a: StageAOutput) -> str:
    for item in stage_a.evidence_items:
        if item.evidence_type == "global_text":
            return item.text
    return ""


def _stage_a_multimodal_relation(stage_a: StageAOutput) -> str:
    """Read the canonical Stage A relation cue with legacy alias fallback."""

    aux_labels = stage_a.metadata.auxiliary_labels if hasattr(stage_a.metadata, "auxiliary_labels") else {}
    if not isinstance(aux_labels, dict):
        return ""
    return str(
        aux_labels.get("stage_a_multimodal_relation")
        or aux_labels.get("multimodal_relation")
        or ""
    )


def _harmfulness_payload(prediction: Prediction) -> dict[str, object]:
    return {
        "label": prediction.label,
        "score": prediction.score,
        "scores": prediction.scores,
        "logits": prediction.logits,
        "decision_threshold": 0.5,
    }


def _target_payload(
    prediction: Prediction,
    stage_a: StageAOutput,
    presence_prediction: Prediction | None = None,
) -> dict[str, object]:
    text = _ocr_text(stage_a)
    heuristic_presence_score = target_presence_score(text)
    entities = capitalized_spans(text, limit=5)
    lowered = text.lower()
    attributes = []
    attribute_terms = {
        "political_ideology": ["trump", "obama", "bernie", "election", "brexit", "debate", "party"],
        "nationality": ["america", "china", "thais", "european", "hong kong"],
        "sex_gender": ["women", "men", "real men"],
        "religion": ["muslim", "jew", "christian"],
        "race_ethnicity": ["black", "white", "asian"],
    }
    for label, terms in attribute_terms.items():
        if any(term in lowered for term in terms):
            attributes.append(label)
    heuristic_presence = (
        "explicit"
        if entities or heuristic_presence_score >= 0.55
        else "implicit"
        if heuristic_presence_score >= 0.25
        else "none"
    )
    presence = presence_prediction.label if presence_prediction is not None else heuristic_presence
    presence_score = presence_prediction.score if presence_prediction is not None else heuristic_presence_score
    return {
        "label": prediction.label,
        "score": prediction.score,
        "scores": prediction.scores,
        "logits": prediction.logits,
        "presence": presence,
        "presence_score": presence_score,
        "presence_scores": presence_prediction.scores if presence_prediction is not None else {},
        "presence_logits": presence_prediction.logits if presence_prediction is not None else None,
        "presence_source": "target_presence_head" if presence_prediction is not None else "heuristic_target_presence_score",
        "presence_provenance": "logits_aux" if presence_prediction is not None else "heuristic_proxy",
        "heuristic_presence": heuristic_presence,
        "heuristic_presence_score": heuristic_presence_score,
        "heuristic_presence_source": "heuristic_target_presence_score",
        "granularity": prediction.label,
        "attributes": attributes or ["none"],
        "label_summary": ", ".join(entities[:3]) if entities else "",
    }


def _intent_payload(prediction: Prediction, harmfulness: Prediction, stage_a: StageAOutput, stage_c: StageCOutput) -> dict[str, object]:
    scores = sorted(prediction.scores.items(), key=lambda item: item[1], reverse=True)
    secondary = [label for label, _ in scores[1:3]]
    stance = "hostile" if harmfulness.label == "harmful" or prediction.label in {"ridicule_mockery", "denigration_insult", "harassment_threat"} else "supportive" if prediction.label == "solidarity_support" else "neutral"
    knowledge_need = float(
        stage_a.auxiliary_scores.get(
            "knowledge_need_score",
            stage_a.auxiliary_scores.get("knowledge_need", 0.0),
        )
    )
    external_nonfallback = any(item.source != "fallback" for item in stage_c.verified_items)
    return {
        "label": prediction.label,
        "score": prediction.score,
        "scores": prediction.scores,
        "logits": prediction.logits,
        "primary": prediction.label,
        "stance": stance,
        "secondary": secondary,
        "background_knowledge_needed": knowledge_need >= 0.45 or external_nonfallback,
        "background_knowledge_score": knowledge_need,
        "background_knowledge_source": "stage_a_knowledge_need_score_or_stage_c_external_presence",
        "background_knowledge_provenance": "cue_proxy",
    }


def _tactic_payload(
    prediction: Prediction,
    stage_a: StageAOutput,
    relation_prediction: Prediction | None = None,
) -> dict[str, object]:
    cues = rhetorical_cues(_ocr_text(stage_a))
    stage_a_relation = _stage_a_multimodal_relation(stage_a) or "unknown"
    relation = relation_prediction.label if relation_prediction is not None else stage_a_relation
    rhetorical_labels = sorted(set([prediction.label, *cues.keys()]))
    structural = []
    if relation in {"incongruent", "cross_modal_implication"} or stage_a_relation in {"incongruent", "cross_modal_implication"}:
        structural.append("cross_modal_contrast")
    if any(item.evidence_type == "text_span" for item in stage_a.evidence_items):
        structural.append("caption_text_overlay")
    return {
        "label": prediction.label,
        "score": prediction.score,
        "scores": prediction.scores,
        "logits": prediction.logits,
        "rhetorical": rhetorical_labels,
        "rhetorical_labels": rhetorical_labels,
        "rhetorical_primary": prediction.label,
        "rhetorical_scores": prediction.scores,
        "rhetorical_decoding": "top1_logits_plus_heuristic_cues",
        "rhetorical_prediction_source": "tactic_logits_top1_plus_heuristic_cues",
        "heuristic_rhetorical_cues": sorted(cues.keys()),
        "rhetorical_provenance": "logits_multilabel_or_top1_rendered",
        "multimodal_relation": relation,
        "multimodal_relation_score": relation_prediction.score if relation_prediction is not None else 0.0,
        "multimodal_relation_scores": relation_prediction.scores if relation_prediction is not None else {},
        "multimodal_relation_logits": relation_prediction.logits if relation_prediction is not None else None,
        "multimodal_relation_source": "tactic_multimodal_relation_head" if relation_prediction is not None else "stage_a_cue",
        "multimodal_relation_provenance": "logits_aux" if relation_prediction is not None else "stage_a_cue_proxy",
        "stage_a_multimodal_relation": stage_a_relation,
        "stage_a_multimodal_relation_source": "stage_a_cue",
        "structural": structural or ["single_panel_caption"],
        "structural_tactics": structural or ["single_panel_caption"],
        "keywords": keyword_candidates(_ocr_text(stage_a), limit=8),
    }


def _field_provenance() -> dict[str, str]:
    return {
        "harmfulness.label": "logits",
        "target.granularity": "logits",
        "target.presence": "logits_aux",
        "target.heuristic_presence": "heuristic_proxy",
        "target.attributes": "heuristic_proxy",
        "intent.primary": "logits",
        "intent.stance": "rule_proxy",
        "intent.secondary": "score_rank_proxy",
        "intent.background_knowledge_needed": "cue_proxy",
        "tactic.rhetorical": "logits_multilabel_or_top1_rendered",
        "tactic.multimodal_relation": "logits_aux",
        "tactic.stage_a_multimodal_relation": "stage_a_cue_proxy",
        "tactic.structural": "rule_proxy",
        "supporting_evidence": "gate_attention_score_proxy",
        "rationale": "template",
    }


def _trainable_logits_fields() -> list[str]:
    return [
        "harmfulness.label",
        "target.granularity",
        "target.presence",
        "intent.primary",
        "tactic.rhetorical",
        "tactic.multimodal_relation",
    ]


def _proxy_fields() -> list[str]:
    return [
        "target.attributes",
        "intent.stance",
        "intent.secondary",
        "intent.background_knowledge_needed",
        "tactic.structural",
        "supporting_evidence",
    ]


def _template_fields() -> list[str]:
    return ["rationale"]


def _cue_fields() -> list[str]:
    return [
        "target.heuristic_presence",
        "intent.background_knowledge_needed",
        "tactic.stage_a_multimodal_relation",
    ]


def _label_spaces(
    harm_pred: Prediction,
    target_pred: Prediction,
    intent_pred: Prediction,
    tactic_pred: Prediction,
    target_presence_pred: Prediction,
    tactic_relation_pred: Prediction,
) -> dict[str, list[str]]:
    return {
        "harmfulness": list(harm_pred.labels),
        "target": list(target_pred.labels),
        "target_presence": list(target_presence_pred.labels),
        "intent": list(intent_pred.labels),
        "tactic": list(tactic_pred.labels),
        "tactic_multimodal_relation": list(tactic_relation_pred.labels),
    }


__all__ = [
    "Prediction",
    "StageEMetadata",
    "StageEOutput",
    "HarmfulnessHead",
    "TargetHead",
    "IntentHead",
    "TacticHead",
    "EvidenceAttributionLayer",
    "TemplateRationaleGenerator",
    "StructuredInterpretationHead",
]
