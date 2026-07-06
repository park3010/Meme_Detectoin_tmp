from __future__ import annotations

from module.internal_evidence_extractor import InternalEvidenceExtractor


def test_stage_a_forward_dummy_sample():
    model = InternalEvidenceExtractor()
    output = model(
        {
            "sample_id": "s1",
            "dataset_name": "memotion",
            "image_path": None,
            "ocr_text_full": "SCHOOLS TEACH USELESS STUFF",
        }
    )
    assert output.internal_tokens.size(1) == 256
    assert len(output.evidence_items) >= 3
    assert "knowledge_need" in output.auxiliary_scores
    assert output.metadata.tensor_shapes["internal_tokens"][1] == 256
