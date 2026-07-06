from __future__ import annotations

import pytest
import torch

from experiments.ablation_runner import _empty_stage_b, _empty_stage_c, _minimal_stage_c_from_stage_b
from module.internal_evidence_extractor import InternalEvidenceExtractor
from module.runner import HarmfulMemePipeline


def test_stage_a_outputs_follow_incongruity_device_cpu():
    model = InternalEvidenceExtractor().eval()
    output = model(_sample())
    device = next(model.incongruity.parameters()).device

    for tensor in [
        output.global_visual,
        output.global_text,
        output.patch_tokens,
        output.text_tokens,
        output.roi_tokens,
        output.internal_tokens,
    ]:
        assert tensor.device == device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_stage_a_outputs_follow_incongruity_device_cuda():
    model = InternalEvidenceExtractor().to("cuda").eval()
    output = model(_sample())
    device = next(model.incongruity.parameters()).device

    assert device.type == "cuda"
    for tensor in [
        output.global_visual,
        output.global_text,
        output.patch_tokens,
        output.text_tokens,
        output.roi_tokens,
        output.internal_tokens,
    ]:
        assert tensor.device == device


def test_diagnostic_empty_knowledge_states_use_stage_device():
    pipeline = HarmfulMemePipeline().eval()
    stage_a = pipeline.stage_a(_sample())
    stage_b = _empty_stage_b(stage_a)
    stage_c = _empty_stage_c(stage_a)

    assert stage_b.candidate_tokens.device == stage_a.internal_tokens.device
    assert stage_c.verified_tokens.device == stage_a.internal_tokens.device
    assert stage_c.support_matrix.device == stage_a.internal_tokens.device
    assert stage_c.final_scores.device == stage_a.internal_tokens.device


def test_minimal_stage_c_from_stage_b_uses_candidate_device():
    pipeline = HarmfulMemePipeline().eval()
    stage_a = pipeline.stage_a(_sample())
    stage_b = pipeline.stage_b(stage_a, _sample()["ocr_text_full"])
    stage_c = _minimal_stage_c_from_stage_b(stage_a, stage_b)

    assert stage_c.verified_tokens.device == stage_b.candidate_tokens.device
    assert stage_c.support_matrix.device == stage_b.candidate_tokens.device
    assert stage_c.final_scores.device == stage_b.candidate_tokens.device


def _sample() -> dict:
    return {
        "sample_id": "device_contract",
        "dataset_name": "harm_c",
        "image_path": None,
        "ocr_text_full": "THIS MEME MOCKS A PUBLIC GROUP WITH SARCASM",
        "raw_label": 1,
    }
