"""Modular loss utilities for Phase 2 research experiments.

Expected supervision can come from annotation JSONL fields such as:
- `harmfulness.label` or binary raw labels
- `target.target_granularity`
- `intent.intent_primary`
- `tactic.tactic_rhetorical`
- evidence ids for internal/external attribution, when available

The helpers accept logits when a training script keeps tensors alive, and can
also operate on detached score dictionaries for smoke tests and analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from utils.text_utils import jaccard_similarity, normalize_text


LOSS_PROVENANCE = {
    "harmfulness": "logits",
    "target_granularity": "logits",
    "intent_primary": "logits",
    "tactic_rhetorical": "logits_multilabel",
    "target_presence": "logits_aux_with_proxy_fallback",
    "target_attributes": "proxy_set_overlap",
    "stance": "proxy_rule_score",
    "secondary_intent": "proxy_set_overlap",
    "background_knowledge_needed": "proxy_detached_score",
    "tactic_multimodal_relation": "logits_aux_with_proxy_fallback",
    "evidence": "proxy_selected_score",
    "evidence_text_alignment": "proxy_text_overlap",
    "consistency": "proxy_detached_metadata",
}


def loss_provenance(name: str) -> str:
    """Return the expected gradient provenance for a loss component."""

    return LOSS_PROVENANCE.get(name, "unknown")


def is_differentiable_loss(name: str, loss: torch.Tensor | None = None) -> bool:
    """Return whether a loss is expected to be logits-based and differentiable."""

    expected = loss_provenance(name).startswith("logits")
    if not expected:
        return False
    return bool(loss.requires_grad) if isinstance(loss, torch.Tensor) else True


@dataclass
class StructuredLossConfig:
    """Weights for structured interpretation losses."""

    harmfulness_weight: float = 1.0
    target_weight: float = 0.7
    intent_weight: float = 0.7
    tactic_weight: float = 0.7
    evidence_weight: float = 0.4
    consistency_weight: float = 0.2
    rationale_weight: float = 0.0
    auxiliary_weights: dict[str, float] = field(default_factory=dict)


def classification_loss_from_logits(logits: torch.Tensor, target_index: int) -> torch.Tensor:
    """Cross-entropy for one example or a batch with integer target index."""

    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    target = torch.tensor([target_index], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, target)


def multilabel_loss_from_logits(logits: torch.Tensor, target_indices: list[int]) -> torch.Tensor:
    """Binary cross-entropy for one multi-label example."""

    if logits.dim() != 1:
        logits = logits.squeeze(0)
    target = torch.zeros_like(logits)
    for idx in target_indices:
        if 0 <= idx < target.numel():
            target[idx] = 1.0
    return F.binary_cross_entropy_with_logits(logits, target)


def binary_loss_from_logits(logit: torch.Tensor, target: bool | int | float) -> torch.Tensor:
    """Binary cross-entropy for one scalar logit."""

    target_tensor = torch.tensor(float(target), dtype=logit.dtype, device=logit.device)
    return F.binary_cross_entropy_with_logits(logit.reshape(()), target_tensor)


def classification_loss_from_scores(scores: dict[str, float], target_label: str) -> torch.Tensor:
    """Negative log-likelihood from a detached label->probability dictionary."""

    prob = max(float(scores.get(target_label, 1e-8)), 1e-8)
    return -torch.log(torch.tensor(prob, dtype=torch.float32))


class EvidenceAttributionLoss(nn.Module):
    """Binary cross-entropy over selected evidence ids when supervision exists."""

    def forward(
        self,
        predicted_scores: dict[str, float],
        gold_ids: set[str],
    ) -> torch.Tensor:
        if not predicted_scores or not gold_ids:
            return torch.tensor(0.0)
        losses = []
        for evidence_id, score in predicted_scores.items():
            target = 1.0 if evidence_id in gold_ids else 0.0
            losses.append(F.binary_cross_entropy(torch.tensor(float(score)).clamp(1e-5, 1 - 1e-5), torch.tensor(target)))
        return torch.stack(losses).mean() if losses else torch.tensor(0.0)


def consistency_regularization_loss(regularizer_hooks: dict[str, float]) -> torch.Tensor:
    """Small differentiable-style proxy from Stage D regularizer hooks."""

    if not regularizer_hooks:
        return torch.tensor(0.0)
    token_sparsity = float(regularizer_hooks.get("token_gate_sparsity", 0.5))
    sufficiency = float(regularizer_hooks.get("knowledge_sufficiency", 0.5))
    entropy = float(regularizer_hooks.get("attention_entropy_proxy", 0.0))
    return torch.tensor((token_sparsity - 0.45) ** 2 + max(0.0, 0.35 - sufficiency) ** 2 + 0.1 * entropy)


class StructuredMemeLoss(nn.Module):
    """Aggregate modular losses from a Stage E output and supervision dict."""

    def __init__(self, config: StructuredLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or StructuredLossConfig()
        self.evidence_loss = EvidenceAttributionLoss()

    def forward(self, stage_e_output: Any, supervision: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Compute available losses without requiring every annotation field."""

        supervision = extract_supervision_from_annotation(supervision)
        losses: dict[str, torch.Tensor] = {}
        if "harmfulness" in supervision:
            losses["harmfulness"] = _prediction_loss(stage_e_output.harmfulness, str(supervision["harmfulness"]))
        if "target_granularity" in supervision:
            losses["target_granularity"] = _prediction_loss(stage_e_output.target, str(supervision["target_granularity"]))
        elif "target" in supervision:
            losses["target_granularity"] = _prediction_loss(stage_e_output.target, str(supervision["target"]))
        if "intent_primary" in supervision:
            losses["intent_primary"] = _prediction_loss(stage_e_output.intent, str(supervision["intent_primary"]))
        elif "intent" in supervision:
            losses["intent_primary"] = _prediction_loss(stage_e_output.intent, str(supervision["intent"]))
        if "tactic_rhetorical" in supervision:
            values = _as_list(supervision["tactic_rhetorical"])
            losses["tactic_rhetorical"] = _multilabel_or_single_prediction_loss(stage_e_output.tactic, values)
        elif "tactic" in supervision:
            values = _as_list(supervision["tactic"])
            losses["tactic_rhetorical"] = _multilabel_or_single_prediction_loss(stage_e_output.tactic, values)

        structured = getattr(stage_e_output, "structured_prediction", {}) or {}
        if "target_presence" in supervision:
            target_presence = str(supervision["target_presence"])
            target_payload = structured.get("target", {})
            if target_presence not in {"ambiguous", "unknown"}:
                losses["target_presence"] = _auxiliary_classification_or_fallback(
                    payload=target_payload,
                    target_label=target_presence,
                    logits_key="presence_logits",
                    label_space=(
                        _structured_label_space(structured, "target_presence")
                        or list((target_payload.get("presence_scores", {}) or {}).keys())
                    ),
                    fallback=lambda: _detached_binary_field_loss(
                        float(target_payload.get("heuristic_presence_score", target_payload.get("presence_score", 0.0))),
                        target_presence != "none",
                    ),
                )
        if "target_attributes" in supervision:
            losses["target_attributes"] = _set_overlap_loss(
                structured.get("target", {}).get("attributes", []),
                _as_list(supervision["target_attributes"]),
            )
        if "stance" in supervision:
            losses["stance"] = classification_loss_from_scores(_stance_scores(structured.get("intent", {})), str(supervision["stance"]))
        if "secondary_intent" in supervision and supervision["secondary_intent"]:
            losses["secondary_intent"] = _set_overlap_loss(
                structured.get("intent", {}).get("secondary", []),
                _as_list(supervision["secondary_intent"]),
            )
        if "background_knowledge_needed" in supervision:
            losses["background_knowledge_needed"] = _detached_binary_field_loss(
                float(structured.get("intent", {}).get("background_knowledge_score", 0.0)),
                bool(supervision["background_knowledge_needed"]),
            )
        if "tactic_multimodal_relation" in supervision:
            tactic_relation = str(supervision["tactic_multimodal_relation"])
            tactic_payload = structured.get("tactic", {})
            if tactic_relation not in {"ambiguous", "unknown"}:
                losses["tactic_multimodal_relation"] = _auxiliary_classification_or_fallback(
                    payload=tactic_payload,
                    target_label=tactic_relation,
                    logits_key="multimodal_relation_logits",
                    label_space=(
                        _structured_label_space(structured, "tactic_multimodal_relation")
                        or list((tactic_payload.get("multimodal_relation_scores", {}) or {}).keys())
                    ),
                    fallback=lambda: _label_match_loss(
                        str(tactic_payload.get("multimodal_relation", "")),
                        tactic_relation,
                    ),
                )

        gold_evidence = set(str(item) for item in supervision.get("evidence_ids", []))
        if gold_evidence:
            predicted = {
                str(item.get("evidence_id") or item.get("knowledge_id")): float(item.get("score", 0.0))
                for group in stage_e_output.supporting_evidence.values()
                for item in group
            }
            losses["evidence"] = self.evidence_loss(predicted, gold_evidence)
        elif supervision.get("evidence_text"):
            losses["evidence_text_alignment"] = _evidence_text_alignment_loss(stage_e_output, _as_list(supervision["evidence_text"]))

        hooks = structured.get("training_hooks", {}).get("stage_d_regularizers", {})
        losses["consistency"] = consistency_regularization_loss(hooks)

        weighted_total = _zero_like(losses)
        weights = {
            "harmfulness": self.config.harmfulness_weight,
            "target_granularity": self.config.target_weight,
            "target_presence": self.config.target_weight,
            "target_attributes": self.config.target_weight,
            "intent_primary": self.config.intent_weight,
            "stance": self.config.intent_weight,
            "secondary_intent": self.config.intent_weight,
            "background_knowledge_needed": self.config.intent_weight,
            "tactic_rhetorical": self.config.tactic_weight,
            "tactic_multimodal_relation": self.config.tactic_weight,
            "evidence": self.config.evidence_weight,
            "evidence_text_alignment": self.config.evidence_weight,
            "consistency": self.config.consistency_weight,
        }
        for name, loss in losses.items():
            weighted_total = weighted_total + loss * weights.get(name, self.config.auxiliary_weights.get(name, 1.0))
        losses["total"] = weighted_total
        return losses

    def describe_losses(self, losses: dict[str, torch.Tensor]) -> dict[str, dict[str, object]]:
        """Return detached values and provenance metadata for logging only."""

        descriptions: dict[str, dict[str, object]] = {}
        for name, loss in losses.items():
            if not isinstance(loss, torch.Tensor):
                continue
            descriptions[name] = {
                "value": float(loss.detach().cpu()),
                "provenance": loss_provenance(name),
                "differentiable_expected": loss_provenance(name).startswith("logits"),
                "requires_grad": bool(loss.requires_grad),
            }
        return descriptions


def extract_supervision_from_annotation(sample_or_annotation: dict[str, Any]) -> dict[str, Any]:
    """Flatten the nested annotation schema into optional supervision fields."""

    if not sample_or_annotation:
        return {}
    normalized = _extract_supervision_from_normalized_targets(sample_or_annotation)
    if normalized:
        return normalized
    annotation = sample_or_annotation.get("annotation") if isinstance(sample_or_annotation.get("annotation"), dict) else sample_or_annotation
    supervision: dict[str, Any] = {}
    flat_keys = {
        "harmfulness",
        "target",
        "target_presence",
        "target_granularity",
        "target_attributes",
        "intent",
        "intent_primary",
        "stance",
        "secondary_intent",
        "background_knowledge_needed",
        "tactic",
        "tactic_rhetorical",
        "tactic_multimodal_relation",
        "evidence_ids",
        "evidence_text",
        "sample_weight",
    }
    for key in flat_keys:
        if key in sample_or_annotation:
            supervision[key] = sample_or_annotation[key]
    raw_label = sample_or_annotation.get("raw_label") if isinstance(sample_or_annotation, dict) else None
    if raw_label is not None:
        supervision["harmfulness"] = "harmful" if str(raw_label) in {"1", "harmful", "true", "True"} else "non_harmful"
    if isinstance(annotation.get("harmfulness"), dict):
        label = annotation["harmfulness"].get("label")
        if label:
            supervision["harmfulness"] = label
    target = annotation.get("target", {}) if isinstance(annotation.get("target"), dict) else {}
    if target:
        _copy_if_present(supervision, target, "target_presence", "target_presence")
        if "target_presence" not in supervision:
            _copy_if_present(supervision, target, "presence", "target_presence")
        _copy_if_present(supervision, target, "target_granularity", "target_granularity")
        if "target_granularity" not in supervision:
            _copy_if_present(supervision, target, "granularity", "target_granularity")
        _copy_if_present(supervision, target, "protected_attribute", "target_attributes")
        if "target_attributes" not in supervision:
            _copy_if_present(supervision, target, "attributes", "target_attributes")
    intent = annotation.get("intent", {}) if isinstance(annotation.get("intent"), dict) else {}
    if intent:
        _copy_if_present(supervision, intent, "intent_primary", "intent_primary")
        _copy_if_present(supervision, intent, "stance", "stance")
        _copy_if_present(supervision, intent, "secondary_intent", "secondary_intent")
        _copy_if_present(supervision, intent, "background_knowledge_needed", "background_knowledge_needed")
    tactic = annotation.get("tactic", {}) if isinstance(annotation.get("tactic"), dict) else {}
    if tactic:
        _copy_if_present(supervision, tactic, "tactic_rhetorical", "tactic_rhetorical")
        _copy_if_present(supervision, tactic, "tactic_multimodal_relation", "tactic_multimodal_relation")
        if "tactic_multimodal_relation" not in supervision:
            _copy_if_present(supervision, tactic, "multimodal_relation", "tactic_multimodal_relation")
    evidence = annotation.get("evidence", {}) if isinstance(annotation.get("evidence"), dict) else {}
    evidence_text = [
        evidence.get("key_text_evidence", ""),
        evidence.get("key_visual_evidence", ""),
        evidence.get("key_cross_modal_evidence", ""),
    ]
    evidence_text = [normalize_text(item) for item in evidence_text if normalize_text(item)]
    if evidence_text:
        supervision["evidence_text"] = evidence_text
    if "sample_weight" not in supervision:
        supervision["sample_weight"] = 1.0
    return supervision


def _extract_supervision_from_normalized_targets(sample: dict[str, Any]) -> dict[str, Any]:
    """Build supervision from NormalizedLabelAdapter fields when available."""

    targets = sample.get("targets") or {}
    label_strings = targets.get("label_strings") or sample.get("label_strings") or {}
    if not isinstance(label_strings, dict) or not label_strings:
        return {}
    masks = targets.get("masks") or {}
    supervision: dict[str, Any] = {}
    field_map = {
        "harmfulness": "harmfulness",
        "target_presence": "target_presence",
        "target_granularity": "target_granularity",
        "protected_attribute": "target_attributes",
        "intent_primary": "intent_primary",
        "secondary_intent": "secondary_intent",
        "stance": "stance",
        "background_knowledge_needed": "background_knowledge_needed",
        "tactic_rhetorical": "tactic_rhetorical",
        "tactic_multimodal_relation": "tactic_multimodal_relation",
    }
    for source_key, out_key in field_map.items():
        if source_key not in label_strings:
            continue
        if int(masks.get(source_key, 1) or 0) == 0:
            continue
        value = label_strings[source_key]
        if value is None or value == "":
            continue
        supervision[out_key] = value

    evidence_text = _collect_normalized_evidence_text(sample, targets)
    if evidence_text:
        supervision["evidence_text"] = evidence_text
    if "evidence_ids" in sample:
        supervision["evidence_ids"] = sample["evidence_ids"]
    supervision["sample_weight"] = _normalized_sample_weight(sample, targets)
    return supervision


def _collect_normalized_evidence_text(sample: dict[str, Any], targets: dict[str, Any]) -> list[str]:
    # TODO: Add an experiment switch for strict evidence vs expanded context.
    # Strict evidence should include key_text_evidence, key_visual_evidence,
    # key_cross_modal_evidence, target_text_span, target_visual_cue,
    # tactic_text_span, and tactic_image_region_description. Expanded context
    # should separately include visual_scene_summary, surface_multimodal_relation,
    # intent_free_text, and background_knowledge_text.
    evidence = sample.get("evidence_text") or targets.get("evidence_text") or {}
    if isinstance(evidence, list):
        return [normalize_text(item) for item in evidence if normalize_text(item)]
    if not isinstance(evidence, dict):
        return []
    keys = [
        "visual_scene_summary",
        "surface_multimodal_relation",
        "target_text_span",
        "target_visual_cue",
        "intent_free_text",
        "background_knowledge_text",
        "tactic_text_span",
        "tactic_image_region_description",
        "key_text_evidence",
        "key_visual_evidence",
        "key_cross_modal_evidence",
    ]
    texts: list[str] = []
    for key in keys:
        text = normalize_text(evidence.get(key, ""))
        if text:
            texts.append(text)
    return texts


def _normalized_sample_weight(sample: dict[str, Any], targets: dict[str, Any]) -> float:
    value = sample.get("sample_weight", targets.get("sample_weight", 1.0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0


def _prediction_loss(prediction: Any, target_label: str) -> torch.Tensor:
    labels = list(getattr(prediction, "labels", []) or getattr(prediction, "scores", {}).keys())
    if labels and target_label in labels and getattr(prediction, "logits", None) is not None:
        return classification_loss_from_logits(prediction.logits, labels.index(target_label))
    return classification_loss_from_scores(getattr(prediction, "scores", {}), target_label)


def _multilabel_or_single_prediction_loss(prediction: Any, target_labels: list[str]) -> torch.Tensor:
    labels = list(getattr(prediction, "labels", []) or getattr(prediction, "scores", {}).keys())
    logits = getattr(prediction, "logits", None)
    if labels and logits is not None:
        indices = [labels.index(label) for label in target_labels if label in labels]
        if indices:
            return multilabel_loss_from_logits(logits, indices)
    target = target_labels[0] if target_labels else ""
    return classification_loss_from_scores(getattr(prediction, "scores", {}), target)


def _detached_binary_field_loss(score: float, target: bool) -> torch.Tensor:
    score_tensor = torch.tensor(score, dtype=torch.float32).clamp(1e-6, 1 - 1e-6)
    target_tensor = torch.tensor(float(target), dtype=torch.float32)
    return F.binary_cross_entropy(score_tensor, target_tensor)


def _set_overlap_loss(predicted: Any, gold: list[str]) -> torch.Tensor:
    predicted_set = {str(item) for item in _as_list(predicted)}
    gold_set = {str(item) for item in gold if item not in {"", "none", None}}
    if not gold_set:
        return torch.tensor(0.0)
    overlap = len(predicted_set & gold_set) / max(1, len(gold_set))
    return torch.tensor(1.0 - overlap, dtype=torch.float32)


def _label_match_loss(predicted: str, gold: str) -> torch.Tensor:
    if not gold:
        return torch.tensor(0.0)
    return torch.tensor(0.0 if predicted == gold else 1.0, dtype=torch.float32)


def _structured_label_space(structured: dict[str, Any], field: str) -> list[str]:
    """Read an auxiliary label space from Stage E training hooks or provenance."""

    hooks = structured.get("training_hooks", {}) or {}
    label_spaces = hooks.get("label_spaces", {}) or {}
    if not label_spaces:
        label_spaces = (structured.get("output_provenance", {}) or {}).get("label_spaces", {}) or {}
    return [str(label) for label in label_spaces.get(field, [])]


def _auxiliary_classification_or_fallback(
    payload: dict[str, Any],
    target_label: str,
    logits_key: str,
    label_space: list[str],
    fallback: Any,
) -> torch.Tensor:
    """Prefer live auxiliary logits, falling back to the legacy proxy loss."""

    logits = payload.get(logits_key)
    if isinstance(logits, torch.Tensor) and target_label in label_space and logits.numel() == len(label_space):
        return classification_loss_from_logits(logits, label_space.index(target_label))
    return fallback()


def _stance_scores(intent_payload: dict[str, Any]) -> dict[str, float]:
    stance = str(intent_payload.get("stance", "neutral"))
    return {label: 0.8 if label == stance else 0.1 for label in ["supportive", "neutral", "hostile"]}


def _evidence_text_alignment_loss(stage_e_output: Any, gold_texts: list[str]) -> torch.Tensor:
    predicted_texts = [
        str(item.get("text", ""))
        for group in getattr(stage_e_output, "supporting_evidence", {}).values()
        for item in group
    ]
    if not predicted_texts or not gold_texts:
        return torch.tensor(0.0)
    best_scores = []
    for gold in gold_texts:
        best_scores.append(max(jaccard_similarity(gold, pred) for pred in predicted_texts))
    return torch.tensor(1.0 - sum(best_scores) / max(1, len(best_scores)), dtype=torch.float32)


def _zero_like(losses: dict[str, torch.Tensor]) -> torch.Tensor:
    for loss in losses.values():
        return loss.new_tensor(0.0)
    return torch.tensor(0.0)


def _copy_if_present(output: dict[str, Any], source: dict[str, Any], key: str, out_key: str) -> None:
    if key not in source:
        return
    value = source[key]
    if value is None or value == "":
        return
    output[out_key] = value


def _as_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]
