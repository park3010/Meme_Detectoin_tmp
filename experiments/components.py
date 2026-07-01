"""Debug helpers for tracing pipeline implementation components."""

from __future__ import annotations

from typing import Any


CANONICAL_STAGE_PATHS = {
    "stage_a": "module.internal_evidence_extractor.InternalEvidenceExtractor",
    "stage_b": "module.external_knowledge_acquisition.ExternalKnowledgeAcquisition",
    "stage_c": "module.knowledge_filter_verifier.KnowledgeRelevanceFilterVerifier",
    "stage_d": "module.evidence_fusion_reasoning.EvidenceFusionReasoning",
    "stage_e": "module.structured_interpretation_head.StructuredInterpretationHead",
}


def print_pipeline_components(model: Any) -> None:
    """Print the main classes/backends used by a HarmfulMemePipeline instance."""

    print("Pipeline components:")
    for attr, canonical in CANONICAL_STAGE_PATHS.items():
        stage = getattr(model, attr, None)
        actual = _class_path(stage)
        print(f"- {attr.replace('_', ' ').title()}: {canonical} (actual: {actual})")
    stage_a = getattr(model, "stage_a", None)
    stage_b = getattr(model, "stage_b", None)
    if stage_a is not None:
        visual = getattr(getattr(stage_a, "visual_encoder", None), "clip", None)
        text = getattr(getattr(stage_a, "text_encoder", None), "encoder", None)
        detector = getattr(getattr(stage_a, "local_extractor", None), "detector", None)
        print(f"- CLIP/image backend: {_backend(visual)}")
        print(f"- Text encoder backend: {_backend(text)}")
        print(f"- Detector mode: {getattr(detector, 'mode', 'unknown')}")
    if stage_b is not None:
        retriever = getattr(getattr(stage_b, "retriever", None), "adapter", None)
        generator = getattr(stage_b, "generator", None)
        print(f"- Retriever backend: {retriever.__class__.__module__}.{retriever.__class__.__name__}" if retriever else "- Retriever backend: unknown")
        print(f"- Generator: {generator.__class__.__module__}.{generator.__class__.__name__}" if generator else "- Generator: unknown")


def _class_path(obj: Any) -> str:
    if obj is None:
        return "missing"
    cls = obj.__class__
    return f"{cls.__module__}.{cls.__name__}"


def _backend(obj: Any) -> str:
    if obj is None:
        return "unknown"
    backend = getattr(obj, "backend", "unknown")
    model_name = getattr(obj, "model_name", None)
    fallback = backend in {"fallback", "hashing"}
    suffix = f", model={model_name}" if model_name else ""
    return f"{backend}, fallback={fallback}{suffix}"
