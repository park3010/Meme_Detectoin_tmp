"""Reusable research losses for the modular meme framework."""

from module.losses.structured_losses import (
    EvidenceAttributionLoss,
    LOSS_PROVENANCE,
    StructuredLossConfig,
    StructuredMemeLoss,
    classification_loss_from_logits,
    classification_loss_from_scores,
    binary_loss_from_logits,
    consistency_regularization_loss,
    extract_supervision_from_annotation,
    is_differentiable_loss,
    loss_provenance,
    multilabel_loss_from_logits,
)

__all__ = [
    "EvidenceAttributionLoss",
    "LOSS_PROVENANCE",
    "StructuredLossConfig",
    "StructuredMemeLoss",
    "binary_loss_from_logits",
    "classification_loss_from_logits",
    "classification_loss_from_scores",
    "consistency_regularization_loss",
    "extract_supervision_from_annotation",
    "is_differentiable_loss",
    "loss_provenance",
    "multilabel_loss_from_logits",
]
