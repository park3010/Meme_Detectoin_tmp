"""Stage C orchestration module."""

from __future__ import annotations

import torch
from torch import nn

from module.stage_a.schemas import StageAOutput
from module.stage_b.schemas import KnowledgeCandidate, StageBOutput
from module.stage_c.redundancy_reducer import RedundancyReducer
from module.stage_c.relevance_scorer import EvidenceAwareRelevanceScorer
from module.stage_c.schemas import StageCMetadata, StageCOutput, VerifiedKnowledgeItem
from module.stage_c.support_verifier import SupportContradictionVerifier
from module.stage_c.validator import CredibilityTemporalValidator


SUPPORT_MATRIX_COLUMNS = ["relevance", "target_support", "intent_support", "tactic_support", "validity", "final"]
FINAL_SCORE_WEIGHTS = {
    "relevance": 0.46,
    "support": 0.24,
    "validity": 0.20,
    "retrieval_prior": 0.10,
}
SUPPORT_LABEL_BONUS = {
    "support": 0.10,
    "insufficient": 0.02,
    "contradict": -0.04,
}


class KnowledgeRelevanceFilterVerifier(nn.Module):
    """Filter and verify external knowledge candidates."""

    def __init__(
        self,
        hidden_dim: int = 256,
        top_k: int = 8,
        min_relevance: float = 0.05,
        allow_low_relevance_fallback: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.min_relevance = min_relevance
        self.allow_low_relevance_fallback = allow_low_relevance_fallback
        self.relevance = EvidenceAwareRelevanceScorer(hidden_dim=hidden_dim)
        self.verifier = SupportContradictionVerifier()
        self.validator = CredibilityTemporalValidator()
        self.reducer = RedundancyReducer()

    def forward(self, stage_a: StageAOutput, stage_b: StageBOutput) -> StageCOutput:
        """Run relevance filtering, verification, validation, and redundancy reduction."""

        internal_summary = self.relevance.summarize_internal(stage_a)
        device = next(self.parameters()).device
        internal_vector = (
            stage_a.internal_tokens.float().mean(dim=0).to(device)
            if stage_a.internal_tokens.numel()
            else torch.zeros(self.hidden_dim, device=device)
        )
        claims = _build_claims(stage_a)
        provisional: list[VerifiedKnowledgeItem] = []
        token_by_id: dict[str, torch.Tensor] = {}
        support_rows_by_id: dict[str, list[float]] = {}
        rejection_records: list[dict[str, object]] = []

        for candidate in stage_b.knowledge_candidates:
            candidate_vector = _candidate_vector(stage_b, candidate, self.hidden_dim)
            token_by_id[candidate.candidate_id] = candidate_vector
            relevance_score, pair_feature, relevance_components = self.relevance.score_pair(
                internal_vector,
                candidate_vector,
                internal_summary,
                candidate,
            )
            is_low_relevance_fallback = candidate.source == "fallback" and self.allow_low_relevance_fallback
            if relevance_score < self.min_relevance and not is_low_relevance_fallback:
                rejection_records.append(_low_relevance_rejection(candidate, relevance_score, self))
                continue
            claim_support = self.verifier.verify_claims(claims, candidate.text)
            support_scores = [float(claim_support[name]["score"]) for name in ["target", "intent", "tactic"]]
            support_labels = [str(claim_support[name]["label"]) for name in ["target", "intent", "tactic"]]
            support_label = _aggregate_support_label(support_labels)
            support_score = sum(support_scores) / max(1, len(support_scores))
            validity_score, validity_components = self.validator.validate(candidate, internal_summary)
            label_bonus = SUPPORT_LABEL_BONUS.get(support_label, 0.0)
            final_components = {
                "relevance": FINAL_SCORE_WEIGHTS["relevance"] * relevance_score,
                "support": FINAL_SCORE_WEIGHTS["support"] * support_score,
                "validity": FINAL_SCORE_WEIGHTS["validity"] * validity_score,
                "retrieval_prior": FINAL_SCORE_WEIGHTS["retrieval_prior"] * candidate.score,
                "label_bonus": label_bonus,
            }
            final_score = max(0.0, min(1.0, sum(final_components.values())))
            support_label_by_claim = {name: str(claim_support[name]["label"]) for name in ["target", "intent", "tactic"]}
            provisional.append(
                VerifiedKnowledgeItem(
                    knowledge_id=candidate.candidate_id,
                    text=candidate.text,
                    source=candidate.source,
                    relevance_score=float(relevance_score),
                    support_label=support_label,
                    support_score=float(support_score),
                    validity_score=float(validity_score),
                    final_score=float(final_score),
                    metadata={
                        "candidate": candidate.metadata,
                        "validity": validity_components,
                        "query": candidate.query,
                        "claim_support": claim_support,
                        "support_labels": support_label_by_claim,
                        "relevance": relevance_components,
                        "validity_components": validity_components,
                        "final_score_components": final_components,
                        "raw_score_components": {
                            "relevance": float(relevance_score),
                            "support": float(support_score),
                            "validity": float(validity_score),
                            "retrieval_prior": float(candidate.score),
                        },
                        "pair_feature_shape": list(pair_feature.shape),
                        "candidate_origin": _candidate_origin(candidate),
                        "is_external_knowledge": bool(candidate.metadata.get("is_external_knowledge", False)),
                        "is_generated": bool(candidate.metadata.get("is_generated", candidate.candidate_type == "generated_hypothesis")),
                        "is_fallback": bool(candidate.metadata.get("is_fallback", candidate.source == "fallback")),
                        "is_retrieved": bool(
                            candidate.metadata.get(
                                "is_retrieved",
                                candidate.candidate_type == "retrieved" and candidate.source != "fallback",
                            )
                        ),
                        "requires_verification": bool(candidate.metadata.get("requires_verification", True)),
                        "verification_status": "accepted",
                        "score_policy": {
                            "weights": dict(FINAL_SCORE_WEIGHTS),
                            "label_bonus": dict(SUPPORT_LABEL_BONUS),
                        },
                    },
                )
            )
            support_rows_by_id[candidate.candidate_id] = [
                float(relevance_score),
                *support_scores,
                float(validity_score),
                float(final_score),
            ]

        verified = self.reducer.reduce_with_tokens(provisional, token_by_id, top_k=self.top_k)
        token_rows = []
        for idx, item in enumerate(verified):
            item.token_index = idx
            token_rows.append(token_by_id.get(item.knowledge_id, torch.zeros(self.hidden_dim, device=device)).to(device))

        verified_tokens = torch.stack(token_rows, dim=0) if token_rows else torch.zeros(0, self.hidden_dim, device=device)
        support_rows = [support_rows_by_id.get(item.knowledge_id, [0.0] * 6) for item in verified]
        support_matrix = torch.tensor(support_rows, dtype=torch.float32, device=device) if support_rows else torch.zeros(0, 6, device=device)
        final_scores = torch.tensor([item.final_score for item in verified], dtype=torch.float32, device=device)
        return StageCOutput(
            sample_id=stage_a.sample_id,
            dataset_name=stage_a.dataset_name,
            verified_items=verified,
            verified_tokens=verified_tokens,
            support_matrix=support_matrix,
            final_scores=final_scores,
            internal_summary=internal_summary,
            metadata=StageCMetadata(
                input_candidate_count=len(stage_b.knowledge_candidates),
                filtered_candidate_count=len(verified),
                top_k=self.top_k,
                min_relevance=self.min_relevance,
                claim_types=["target", "intent", "tactic"],
                score_fields=SUPPORT_MATRIX_COLUMNS,
                support_matrix_columns=SUPPORT_MATRIX_COLUMNS,
                allow_low_relevance_fallback=self.allow_low_relevance_fallback,
                input_origin_counts=_origin_counts(stage_b.knowledge_candidates),
                verified_origin_counts=_verified_origin_counts(verified),
                rejected_count=len(rejection_records),
                rejection_records=rejection_records,
                score_weights=dict(FINAL_SCORE_WEIGHTS),
                label_bonus=dict(SUPPORT_LABEL_BONUS),
                verification_policy={
                    "policy_type": "lightweight_candidate_filter",
                    "requires_stage_c_verification": True,
                    "min_relevance": float(self.min_relevance),
                    "allow_low_relevance_fallback": bool(self.allow_low_relevance_fallback),
                    "support_labels": ["support", "contradict", "insufficient"],
                    "support_matrix_columns": list(SUPPORT_MATRIX_COLUMNS),
                },
            ),
        )


def _find_candidate_index(candidates: list[KnowledgeCandidate], candidate_id: str) -> int | None:
    for idx, candidate in enumerate(candidates):
        if candidate.candidate_id == candidate_id:
            return idx
    return None


def _candidate_vector(stage_b: StageBOutput, candidate: KnowledgeCandidate, hidden_dim: int) -> torch.Tensor:
    if 0 <= candidate.token_index < stage_b.candidate_tokens.size(0):
        return stage_b.candidate_tokens[candidate.token_index].float()
    source_idx = _find_candidate_index(stage_b.knowledge_candidates, candidate.candidate_id)
    if source_idx is not None and source_idx < stage_b.candidate_tokens.size(0):
        return stage_b.candidate_tokens[source_idx].float()
    device = stage_b.candidate_tokens.device if stage_b.candidate_tokens.numel() else torch.device("cpu")
    return torch.zeros(hidden_dim, device=device)


def _build_claims(stage_a: StageAOutput) -> dict[str, str]:
    relation = ""
    cues: dict[str, float] = {}
    if hasattr(stage_a.metadata, "auxiliary_labels"):
        aux_labels = stage_a.metadata.auxiliary_labels
        relation = str(
            aux_labels.get("stage_a_multimodal_relation")
            or aux_labels.get("multimodal_relation")
            or ""
        )
        raw_cues = aux_labels.get("rhetorical_cues", {})
        cues = raw_cues if isinstance(raw_cues, dict) else {}
    text_spans = [item.text for item in stage_a.evidence_items if item.evidence_type in {"text_span", "global_text"} and item.text]
    local_symbols = [item.text for item in stage_a.evidence_items if item.evidence_type == "local_symbol" and item.text]
    return {
        "target": "Possible target or entity cues: " + "; ".join([*local_symbols[:3], *text_spans[:2]]),
        "intent": "Possible intent cues: " + ", ".join(cues.keys()) + "; " + " ".join(text_spans[:1]),
        "tactic": f"Possible multimodal tactic: {relation}; rhetorical cues: {', '.join(cues.keys())}",
    }


def _aggregate_support_label(labels: list[str]) -> str:
    if labels.count("contradict") >= 2:
        return "contradict"
    if "support" in labels:
        return "support"
    return "insufficient"


def _candidate_origin(candidate: KnowledgeCandidate) -> str:
    origin = candidate.metadata.get("candidate_origin")
    if origin:
        return str(origin)
    if candidate.candidate_type == "generated_hypothesis":
        return "generated_hypothesis"
    if candidate.source == "fallback":
        return "fallback"
    return "retrieved"


def _origin_counts(candidates: list[KnowledgeCandidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        origin = _candidate_origin(candidate)
        counts[origin] = counts.get(origin, 0) + 1
    return counts


def _verified_origin_counts(items: list[VerifiedKnowledgeItem]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        origin = str(item.metadata.get("candidate_origin", "unknown"))
        counts[origin] = counts.get(origin, 0) + 1
    return counts


def _low_relevance_rejection(
    candidate: KnowledgeCandidate,
    relevance_score: float,
    verifier: KnowledgeRelevanceFilterVerifier,
) -> dict[str, object]:
    return {
        "candidate_id": candidate.candidate_id,
        "candidate_origin": _candidate_origin(candidate),
        "source": candidate.source,
        "candidate_type": candidate.candidate_type,
        "reason": "low_relevance",
        "relevance_score": float(relevance_score),
        "min_relevance": float(verifier.min_relevance),
        "allow_low_relevance_fallback": bool(verifier.allow_low_relevance_fallback),
        "is_fallback": bool(candidate.metadata.get("is_fallback", candidate.source == "fallback")),
        "is_generated": bool(candidate.metadata.get("is_generated", candidate.candidate_type == "generated_hypothesis")),
        "is_external_knowledge": bool(candidate.metadata.get("is_external_knowledge", False)),
    }
