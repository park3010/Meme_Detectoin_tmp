from __future__ import annotations

from experiments.ablation_configs import (
    ABLATION_MODES,
    component_state_for_ablation,
    get_ablation_config,
    get_ablation_contract,
    runtime_config_for_ablation,
)
from experiments.ablation_runner import execute_variant_pipeline
from module.runner import HarmfulMemePipeline


def test_ablation_contracts_cover_supported_modes():
    for name in ABLATION_MODES:
        contract = get_ablation_contract(name)
        assert contract.name == name
        assert isinstance(contract.description, str) and contract.description
        assert isinstance(contract.disabled_components, list)
        assert isinstance(contract.expected_stage_behavior, dict)
        assert contract.expected_active_logits_losses
        assert contract.expected_evidence_mode
        assert contract.expected_knowledge_mode


def test_alias_and_component_state_contracts():
    assert get_ablation_config("w_o_verifier").name == "w_o_support_verifier"
    assert get_ablation_contract("w_o_verifier").name == "w_o_support_verifier"

    assert component_state_for_ablation("w_o_retrieval")["stage_b_retrieval_enabled"] is False
    assert component_state_for_ablation("w_o_retrieval")["stage_b_context_generation_enabled"] is False
    assert component_state_for_ablation("w_o_task_aware_gate")["stage_d_task_aware_gate_enabled"] is False


def test_structured_auxiliary_contract_has_four_trainable_losses():
    contract = get_ablation_contract("w_o_structured_auxiliary")

    assert sorted(contract.expected_active_logits_losses) == [
        "harmfulness",
        "intent_primary",
        "tactic_rhetorical",
        "target_granularity",
    ]
    assert sorted(contract.expected_proxy_or_disabled_losses) == [
        "tactic_multimodal_relation",
        "target_presence",
    ]
    assert component_state_for_ablation("w_o_structured_auxiliary")["stage_e_structured_auxiliary_enabled"] is False


def test_label_only_no_evidence_is_not_in_default_causal_contracts():
    contract = get_ablation_contract("label_only_no_evidence")

    assert contract.supported is False
    assert "evaluation-time" in contract.unsupported_reason


def test_ablation_runtime_metadata_matches_contracts():
    pipeline = HarmfulMemePipeline().eval()
    sample = {
        "sample_id": "s1",
        "dataset_name": "harm_c",
        "image_path": None,
        "ocr_text_full": "THIS TEXT MOCKS A PUBLIC GROUP",
        "raw_label": 1,
    }

    no_retrieval = execute_variant_pipeline(pipeline, sample, ablation=get_ablation_config("w_o_retrieval"))
    assert no_retrieval["stage_d"].metadata.verified_knowledge_count == 0
    assert no_retrieval["stage_b"].metadata.retrieved_count == 0

    no_gate = execute_variant_pipeline(pipeline, sample, ablation=get_ablation_config("w_o_task_aware_gate"))
    assert "shared_gate" in no_gate["stage_d"].metadata.gate_mode


def test_train_time_pipeline_ablation_metadata_matches_contracts():
    pipeline = HarmfulMemePipeline().eval()
    sample = {
        "sample_id": "train_time_ablation",
        "dataset_name": "harm_c",
        "image_path": None,
        "ocr_text_full": "THIS TEXT MOCKS A PUBLIC GROUP",
        "raw_label": 1,
    }

    no_retrieval = pipeline(sample, ablation=runtime_config_for_ablation("w_o_retrieval"))
    assert no_retrieval["stage_b"].metadata.retrieved_count == 0
    assert no_retrieval["stage_b"].candidate_tokens.device == no_retrieval["stage_a"].internal_tokens.device
    assert no_retrieval["stage_d"].metadata.verified_knowledge_count == 0

    no_support = pipeline(sample, ablation=runtime_config_for_ablation("w_o_support_verifier"))
    assert no_support["stage_c"].metadata.verification_policy["support_verifier_enabled"] is False
    if no_support["stage_c"].support_matrix.numel():
        assert float(no_support["stage_c"].support_matrix[:, 1:4].sum()) == 0.0

    no_gate = pipeline(sample, ablation=runtime_config_for_ablation("w_o_task_aware_gate"))
    assert "shared_gate" in no_gate["stage_d"].metadata.gate_mode
    assert no_gate["stage_d"].metadata.analysis_hooks["task_aware_gate_enabled"] == 0.0
    assert no_gate["stage_d"].shared_reasoning_state.requires_grad is True
