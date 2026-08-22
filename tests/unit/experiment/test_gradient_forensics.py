import json
from pathlib import Path

import pytest
import torch

from experiments.gradient_forensics import (
    applicable_blocker,
    audit_optimizer_membership,
    gradient_state,
    select_task_microbatches,
    validate_v2_manifest,
)
from dataset.labels import LabelVocab


def test_active_v2_manifest_selected_and_exclusions_absent():
    report=validate_v2_manifest()
    assert report["passed"] and report["excluded_intersection_count"]==0
    assert report["manifest_sha256"]==report["expected_manifest_sha256"]


def test_v1_bound_manifest_rejected():
    report=validate_v2_manifest("result/splits/harmeme/source_split_seed_42.json")
    assert not report["passed"] and "v1_bound_sanity_manifest" in report["errors"]


def test_grad_none_zero_and_tiny_are_distinct():
    assert gradient_state(None)["status"]=="none"
    assert gradient_state(torch.zeros(2))["status"]=="zero"
    tiny=gradient_state(torch.tensor([1e-30]))
    assert tiny["status"]=="finite_nonzero" and tiny["gradient_norm_scientific"]!="0"


@pytest.mark.parametrize("category",["frozen_expected","inactive_by_contract","fixed_non_trainable_component"])
def test_non_applicable_component_never_blocks(category):
    assert not applicable_blocker(category,None,valid_n=1)


def test_zero_valid_supervision_does_not_block():
    assert not applicable_blocker("mandatory_trainable",None,valid_n=0)


def test_mandatory_missing_zero_and_nonfinite_block():
    assert applicable_blocker("mandatory_trainable",None,valid_n=1)
    assert applicable_blocker("mandatory_trainable",torch.zeros(1),valid_n=1)
    assert applicable_blocker("mandatory_trainable",torch.tensor([float("nan")]),valid_n=1)


def test_optimizer_omission_and_membership_detected():
    model=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.Linear(2,1))
    optimizer=torch.optim.AdamW(model[0].parameters(),lr=1e-4)
    rows=audit_optimizer_membership(model,optimizer)
    assert any(r["optimizer_membership"] for r in rows)
    assert any(not r["optimizer_membership"] for r in rows)


def test_detached_logits_and_loss_are_detectable():
    logits=torch.ones(2,requires_grad=True)
    assert logits.requires_grad and not logits.detach().requires_grad
    loss=logits.sum(); assert loss.grad_fn is not None and loss.detach().grad_fn is None


def test_output_dimension_mismatch_is_detectable():
    vocab=LabelVocab.from_yaml(); logits=torch.zeros(2)
    assert logits.numel()!=vocab.num_classes("harmfulness")


def test_forensics_never_names_scientific_checkpoint():
    source=Path("experiments/gradient_forensics.py").read_text()
    assert "torch.save" not in source and "best_model.pt" not in source and "last_model.pt" not in source


def test_original_failure_is_preserved():
    report=json.loads(Path("result/source_sanity_v2/gradient_check/gradient_report.json").read_text())
    assert report["reason"]=="gradient execution is intentionally unavailable without the requested CUDA path"


def test_relevance_mlp_score_remains_connected():
    from module.knowledge_filter_verifier import EvidenceAwareRelevanceScorer
    from module.external_knowledge_acquisition import KnowledgeCandidate
    scorer=EvidenceAwareRelevanceScorer(hidden_dim=4)
    candidate=KnowledgeCandidate(candidate_id="x",text="context",source="fixture",score=0.5,candidate_type="retrieved",query="q")
    score,_,_=scorer.score_pair(torch.ones(4),torch.ones(4),"summary",candidate)
    assert score.requires_grad
    score.backward()
    assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad) for parameter in scorer.feature_mlp.parameters())


def test_lazy_text_projection_inherits_frozen_state():
    from module.backbone.text import TextEncoderWrapper
    encoder=TextEncoderWrapper(hidden_dim=4,prefer_transformers=False)
    for parameter in encoder.parameters(): parameter.requires_grad=False
    encoder._project_matrix(torch.ones(2,7))
    assert encoder._projection is not None
    assert all(not parameter.requires_grad for parameter in encoder._projection.parameters())


def test_selection_is_deterministic_with_synthetic_dataset():
    class D:
        samples=[{"dataset_name":"harm_c","sample_id":"a","targets":{"label_strings":{"harmfulness":"harmful","target_presence":"explicit","target_granularity":"individual","intent_primary":"criticism","tactic_rhetorical":["satire"],"tactic_multimodal_relation":"complementary"},"masks":{}},"label_strings":{}}]
    # The production extractor needs attached label_strings as emitted by the adapter.
    D.samples[0]["label_strings"]=D.samples[0]["targets"]["label_strings"]
    manifest={"train":[{"sample_key":"harm_c::a"}]}
    first,_=select_task_microbatches(manifest,D(),LabelVocab.from_yaml())
    second,_=select_task_microbatches(manifest,D(),LabelVocab.from_yaml())
    assert {k:v[0]["sample_id"] for k,v in first.items()}=={k:v[0]["sample_id"] for k,v in second.items()}
