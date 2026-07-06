from __future__ import annotations

from module.internal_evidence_extractor import InternalEvidenceExtractor


def test_stage_a_internal_evidence_contract():
    output = InternalEvidenceExtractor()(
        {
            "sample_id": "contract_sample",
            "dataset_name": "harm_c",
            "image_path": None,
            "ocr_text_full": "SCHOOLS TEACH USELESS STUFF BUT NOT HOW TO DO TAXES",
        }
    )

    assert output.internal_tokens.size(0) == len(output.evidence_items)
    assert all(0 <= item.token_index < output.internal_tokens.size(0) for item in output.evidence_items)

    evidence_types = {item.evidence_type for item in output.evidence_items}
    assert {"global_visual", "global_text", "cross_modal_incongruity", "text_span"} <= evidence_types

    for item in output.evidence_items:
        assert item.metadata["source_stage"] == "stage_a"
        assert "modality" in item.metadata
        assert "grounding_type" in item.metadata
        assert "is_heuristic" in item.metadata
        assert isinstance(item.metadata["is_heuristic"], bool)

    by_type = {item.evidence_type: item for item in output.evidence_items}
    assert by_type["global_text"].metadata["modality"] == "text"
    assert by_type["global_visual"].metadata["modality"] == "image"
    assert by_type["cross_modal_incongruity"].metadata["modality"] == "cross_modal"


def test_stage_a_auxiliary_cue_aliases_preserve_old_keys():
    output = InternalEvidenceExtractor()(
        {
            "sample_id": "cue_sample",
            "dataset_name": "memotion",
            "image_path": None,
            "ocr_text_full": "WHO NEEDS CONTEXT? YEAH RIGHT.",
        }
    )

    scores = output.auxiliary_scores
    assert scores["knowledge_need"] == scores["knowledge_need_score"]
    assert scores["target_presence"] == scores["target_presence_score"]

    labels = output.metadata.auxiliary_labels
    assert "multimodal_relation" in labels
    assert "stage_a_multimodal_relation" in labels
    assert labels["multimodal_relation"] == labels["stage_a_multimodal_relation"]
