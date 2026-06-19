"""Stage E orchestration module."""

from __future__ import annotations

import torch
from torch import nn

from module.stage_a.schemas import StageAOutput
from module.stage_c.schemas import StageCOutput
from module.stage_d.schemas import StageDOutput
from module.stage_e.evidence_attribution import EvidenceAttributionLayer
from module.stage_e.harmfulness_head import HarmfulnessHead
from module.stage_e.intent_head import IntentHead
from module.stage_e.rationale_generator import TemplateRationaleGenerator
from module.stage_e.schemas import Prediction, StageEMetadata, StageEOutput
from module.stage_e.tactic_head import TacticHead
from module.stage_e.target_head import TargetHead
from utils.text_utils import capitalized_spans, keyword_candidates, rhetorical_cues, target_presence_score


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
