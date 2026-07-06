from __future__ import annotations

from module.internal_evidence_extractor import InternalEvidenceExtractor
from module.external_knowledge_acquisition import ExternalKnowledgeAcquisition
from module.knowledge_filter_verifier import KnowledgeRelevanceFilterVerifier
from module.evidence_fusion_reasoning import EvidenceFusionReasoning
from module.evidence_fusion_reasoning import EvidenceGate


def test_stage_d_forward_pytorch_outputs():
    stage_a = InternalEvidenceExtractor()({"sample_id": "s1", "dataset_name": "facebook", "image_path": None, "ocr_text_full": "funny meme"})
    stage_b = ExternalKnowledgeAcquisition()(stage_a)
    stage_c = KnowledgeRelevanceFilterVerifier()(stage_a, stage_b)
    output = EvidenceFusionReasoning()(stage_a, stage_c)
    assert output.shared_reasoning_state.shape == (256,)
    assert "target" in output.task_latents
    assert output.metadata.support_matrix_shape
    assert "external_gate" in output.gates


def test_stage_d_gate_accepts_internal_and_fused_memory():
    gate = EvidenceGate()
    internal = __import__("torch").zeros(4, 256)
    fused = __import__("torch").ones(4, 256)
    gated, gates = gate(fused, internal_memory=internal, knowledge_need=0.2)
    assert gated.shape == internal.shape
    assert gates["external_gate"].shape == (4,)


def test_stage_d_zero_verified_knowledge_does_not_crash():
    stage_a = InternalEvidenceExtractor()({"sample_id": "s1", "dataset_name": "facebook", "image_path": None, "ocr_text_full": "tiny"})
    stage_b = ExternalKnowledgeAcquisition(corpus_paths=[], fallback_candidates=True)(stage_a)
    stage_c = KnowledgeRelevanceFilterVerifier(min_relevance=0.99, allow_low_relevance_fallback=False)(stage_a, stage_b)
    output = EvidenceFusionReasoning()(stage_a, stage_c)
    assert output.cross_attention_weights.shape[1] == 0
    assert output.shared_reasoning_state.shape == (256,)
