"""Experiment-level ablation and comparison mode configuration."""

from __future__ import annotations

from dataclasses import dataclass


ABLATION_MODES = [
    "full",
    "w_o_roi",
    "w_o_incongruity",
    "w_o_retrieval",
    "w_o_context_generation",
    "w_o_relevance_scorer",
    "w_o_support_verifier",
    "w_o_temporal_cultural_validator",
    "w_o_task_aware_gate",
    "w_o_structured_auxiliary",
    "label_only_no_evidence",
]

ABLATION_ALIASES = {"w_o_verifier": "w_o_support_verifier"}

KNOWLEDGE_MODES = ["no_knowledge", "generated_only", "retrieved_only", "generated_retrieved", "verified"]

FUSION_MODES = ["concat_mlp", "mean_pooling", "cross_attention", "shared_gate", "task_aware_gate", "task_aware_gate_verified"]


@dataclass
class AblationConfig:
    """Flags controlling experiment-level ablations."""

    name: str = "full"
    remove_roi: bool = False
    remove_incongruity: bool = False
    disable_retrieval: bool = False
    disable_context_generation: bool = False
    disable_relevance_scorer: bool = False
    disable_support_verifier: bool = False
    disable_temporal_cultural_validator: bool = False
    disable_task_aware_gate: bool = False
    disable_structured_auxiliary: bool = False
    label_only_no_evidence: bool = False


def get_ablation_config(name: str) -> AblationConfig:
    """Return an AblationConfig for one supported mode."""

    name = ABLATION_ALIASES.get(name, name)
    if name not in ABLATION_MODES:
        raise ValueError(f"Unsupported ablation: {name}")
    cfg = AblationConfig(name=name)
    if name == "w_o_roi":
        cfg.remove_roi = True
    elif name == "w_o_incongruity":
        cfg.remove_incongruity = True
    elif name == "w_o_retrieval":
        cfg.disable_retrieval = True
    elif name == "w_o_context_generation":
        cfg.disable_context_generation = True
    elif name == "w_o_relevance_scorer":
        cfg.disable_relevance_scorer = True
    elif name == "w_o_support_verifier":
        cfg.disable_support_verifier = True
    elif name == "w_o_temporal_cultural_validator":
        cfg.disable_temporal_cultural_validator = True
    elif name == "w_o_task_aware_gate":
        cfg.disable_task_aware_gate = True
    elif name == "w_o_structured_auxiliary":
        cfg.disable_structured_auxiliary = True
    elif name == "label_only_no_evidence":
        cfg.label_only_no_evidence = True
    return cfg
