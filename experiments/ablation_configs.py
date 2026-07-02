"""Experiment-level ablation and comparison mode configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass


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

LOGITS_LOSSES = [
    "harmfulness",
    "target_granularity",
    "target_presence",
    "intent_primary",
    "tactic_rhetorical",
    "tactic_multimodal_relation",
]

STRUCTURED_AUXILIARY_LOSSES = ["target_presence", "tactic_multimodal_relation"]

COMPONENT_STATE_KEYS = [
    "stage_a_roi_enabled",
    "stage_a_incongruity_enabled",
    "stage_b_retrieval_enabled",
    "stage_b_context_generation_enabled",
    "stage_c_relevance_enabled",
    "stage_c_support_verifier_enabled",
    "stage_c_validity_enabled",
    "stage_d_task_aware_gate_enabled",
    "stage_e_structured_auxiliary_enabled",
]


@dataclass
class AblationContract:
    """Auditable semantic contract for one ablation mode."""

    name: str
    description: str
    disabled_components: list[str]
    expected_stage_behavior: dict[str, str]
    expected_active_logits_losses: list[str]
    expected_proxy_or_disabled_losses: list[str]
    expected_evidence_mode: str
    expected_knowledge_mode: str
    supported: bool = True
    unsupported_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return asdict(self)


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


def normalize_ablation_name(name: str) -> str:
    """Normalize public aliases to canonical ablation names."""

    return ABLATION_ALIASES.get(name, name)


def get_ablation_config(name: str) -> AblationConfig:
    """Return an AblationConfig for one supported mode."""

    name = normalize_ablation_name(name)
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


def default_component_state() -> dict[str, bool]:
    """Return the full-model component-state vector."""

    return {key: True for key in COMPONENT_STATE_KEYS}


def component_state_for_ablation(name: str) -> dict[str, bool]:
    """Return expected on/off component state for one ablation."""

    canonical = normalize_ablation_name(name)
    state = default_component_state()
    if canonical == "w_o_roi":
        state["stage_a_roi_enabled"] = False
    elif canonical == "w_o_incongruity":
        state["stage_a_incongruity_enabled"] = False
    elif canonical == "w_o_retrieval":
        state["stage_b_retrieval_enabled"] = False
        state["stage_b_context_generation_enabled"] = False
    elif canonical == "w_o_context_generation":
        state["stage_b_context_generation_enabled"] = False
    elif canonical == "w_o_relevance_scorer":
        state["stage_c_relevance_enabled"] = False
    elif canonical == "w_o_support_verifier":
        state["stage_c_support_verifier_enabled"] = False
    elif canonical == "w_o_temporal_cultural_validator":
        state["stage_c_validity_enabled"] = False
    elif canonical == "w_o_task_aware_gate":
        state["stage_d_task_aware_gate_enabled"] = False
    elif canonical == "w_o_structured_auxiliary":
        state["stage_e_structured_auxiliary_enabled"] = False
    elif canonical == "label_only_no_evidence":
        # This mode is a rendering/evaluation ablation: the final payload hides
        # evidence and rationale after the model has produced label predictions.
        pass
    return state


def get_ablation_contract(name: str) -> AblationContract:
    """Return the explicit semantic contract for one ablation mode."""

    canonical = normalize_ablation_name(name)
    if canonical not in ABLATION_MODES:
        raise ValueError(f"Unsupported ablation: {name}")
    active = list(LOGITS_LOSSES)
    disabled_losses: list[str] = []
    evidence_mode = "internal_external_evidence"
    knowledge_mode = "verified"
    disabled_components: list[str] = []
    behavior: dict[str, str] = {"pipeline": "full five-stage execution"}
    supported = True
    unsupported_reason = ""

    if canonical == "full":
        description = "Full proposed five-stage framework."
    elif canonical == "w_o_roi":
        description = "Remove local ROI/symbol evidence after Stage A."
        disabled_components = ["stage_a.local_roi_symbol_evidence"]
        behavior["stage_a"] = "ROI tokens and local_symbol evidence are removed before Stage B."
    elif canonical == "w_o_incongruity":
        description = "Zero cross-modal incongruity evidence and knowledge-need signal."
        disabled_components = ["stage_a.cross_modal_incongruity"]
        behavior["stage_a"] = "Incongruity token score and knowledge_need are zeroed."
    elif canonical == "w_o_retrieval":
        description = "Bypass Stage B retrieval and feed empty verified knowledge to Stage D."
        disabled_components = ["stage_b.retrieval", "stage_b.context_generation", "stage_c.verified_knowledge"]
        behavior["stage_b"] = "Stage B returns zero candidates."
        behavior["stage_c"] = "Stage C returns an empty verified knowledge bank."
        evidence_mode = "internal_only"
        knowledge_mode = "no_knowledge"
    elif canonical == "w_o_context_generation":
        description = "Keep retrieved candidates but remove generated context hypotheses."
        disabled_components = ["stage_b.context_generation"]
        behavior["stage_b"] = "generated_hypothesis candidates are removed before Stage C."
        knowledge_mode = "retrieved_only_plus_verifier"
    elif canonical == "w_o_relevance_scorer":
        description = "Disable Stage C relevance thresholding by setting relevance filtering permissive."
        disabled_components = ["stage_c.relevance_scorer"]
        behavior["stage_c"] = "min_relevance is set to 0.0 during filtering."
    elif canonical == "w_o_support_verifier":
        description = "Disable task-specific support/contradiction verification."
        disabled_components = ["stage_c.support_verifier"]
        behavior["stage_c"] = "target/intent/tactic support columns are neutralized to insufficient."
    elif canonical == "w_o_temporal_cultural_validator":
        description = "Disable temporal/cultural/source validity adjustment."
        disabled_components = ["stage_c.temporal_cultural_validator"]
        behavior["stage_c"] = "validity scores are neutralized to 1.0."
    elif canonical == "w_o_task_aware_gate":
        description = "Replace task-aware Stage D gates with a shared gate."
        disabled_components = ["stage_d.task_aware_gate"]
        behavior["stage_d"] = "task gates are tied to a shared scalar after fusion."
    elif canonical == "w_o_structured_auxiliary":
        description = "Train/evaluate harmfulness plus primary heads without auxiliary structured losses."
        disabled_components = ["stage_e.structured_auxiliary_losses"]
        behavior["training"] = "target_presence and tactic_multimodal_relation losses are excluded from total loss."
        active = [item for item in LOGITS_LOSSES if item not in STRUCTURED_AUXILIARY_LOSSES]
        disabled_losses = list(STRUCTURED_AUXILIARY_LOSSES)
    elif canonical == "label_only_no_evidence":
        description = "Render label outputs without evidence attribution or rationale payloads."
        disabled_components = ["stage_e.evidence_rendering", "stage_e.rationale_rendering"]
        behavior["stage_e"] = "supporting_evidence and rationale are cleared after prediction."
        evidence_mode = "label_only_rendering"
        supported = False
        unsupported_reason = (
            "This is an evaluation-time rendering ablation only; Stage C/D evidence can still influence labels. "
            "It is excluded from default suites that require causal no-evidence semantics."
        )
    else:
        description = canonical

    return AblationContract(
        name=canonical,
        description=description,
        disabled_components=disabled_components,
        expected_stage_behavior=behavior,
        expected_active_logits_losses=active,
        expected_proxy_or_disabled_losses=disabled_losses,
        expected_evidence_mode=evidence_mode,
        expected_knowledge_mode=knowledge_mode,
        supported=supported,
        unsupported_reason=unsupported_reason,
    )


def all_ablation_contracts() -> dict[str, AblationContract]:
    """Return all supported ablation contracts keyed by canonical name."""

    return {name: get_ablation_contract(name) for name in ABLATION_MODES}
