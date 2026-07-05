"""Stage C: knowledge relevance filtering and verification."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from module.backbone.retrieval import CrossEncoderAdapter
from module.external_knowledge_acquisition import KnowledgeCandidate, StageBOutput
from module.internal_evidence_extractor import StageAOutput
from utils.tensor_utils import hashed_vector
from utils.text_utils import jaccard_similarity, keyword_candidates, language_hint, normalize_text


# =============================================================================
# Stage C schemas
# =============================================================================

@dataclass
class VerifiedKnowledgeItem:
    """A candidate that survived relevance and validation filtering."""

    knowledge_id: str
    text: str
    source: str
    relevance_score: float
    support_label: str
    support_score: float
    validity_score: float
    final_score: float
    token_index: int = -1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageCInput:
    """Input container for Stage C."""

    stage_a: StageAOutput
    stage_b: StageBOutput
    candidates: list[KnowledgeCandidate] | None = None


@dataclass
class StageCMetadata:
    """Auditable verification/filter state produced before evidence fusion."""

    input_candidate_count: int
    filtered_candidate_count: int
    top_k: int
    min_relevance: float
    claim_types: list[str] = field(default_factory=list)
    score_fields: list[str] = field(default_factory=list)
    support_matrix_columns: list[str] = field(default_factory=list)
    allow_low_relevance_fallback: bool = True
    input_origin_counts: dict[str, int] = field(default_factory=dict)
    verified_origin_counts: dict[str, int] = field(default_factory=dict)
    rejected_count: int = 0
    rejection_records: list[dict[str, Any]] = field(default_factory=list)
    score_weights: dict[str, float] = field(default_factory=dict)
    label_bonus: dict[str, float] = field(default_factory=dict)
    verification_policy: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageCOutput:
    """Output from Stage C."""

    sample_id: str
    dataset_name: str
    verified_items: list[VerifiedKnowledgeItem]
    verified_tokens: torch.Tensor
    support_matrix: torch.Tensor
    final_scores: torch.Tensor
    internal_summary: str
    metadata: StageCMetadata

    @property
    def task_support_matrix(self) -> torch.Tensor:
        """Return target/intent/tactic support columns from the 6-column matrix."""

        if self.support_matrix.numel() == 0:
            return torch.zeros(0, 3, dtype=self.support_matrix.dtype, device=self.support_matrix.device)
        return self.support_matrix[:, 1:4]

    @property
    def named_support_columns(self) -> dict[str, torch.Tensor]:
        """Return support matrix columns keyed by semantic name."""

        columns = self.metadata.support_matrix_columns or [
            "relevance",
            "target_support",
            "intent_support",
            "tactic_support",
            "validity",
            "final",
        ]
        return {name: self.support_matrix[:, idx] for idx, name in enumerate(columns) if idx < self.support_matrix.size(1)}


# =============================================================================
# Relevance scoring
# =============================================================================

class EvidenceAwareRelevanceScorer(nn.Module):
    """Score candidate knowledge with pairwise embedding features."""

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cross_encoder = CrossEncoderAdapter()
        self.feature_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def summarize_internal(self, stage_a: StageAOutput) -> str:
        """Create a compact textual summary of internal evidence."""

        parts = [
            normalize_text(item.text)
            for item in stage_a.evidence_items
            if item.evidence_type in {"global_text", "local_symbol", "cross_modal_incongruity"}
        ]
        return "; ".join(part for part in parts if part)[:600]

    def score(self, internal_summary: str, candidate: KnowledgeCandidate) -> float:
        """Return relevance score using lexical pair scoring and retriever prior."""

        pair_score = self.cross_encoder.score(internal_summary, candidate.text)
        prior = max(0.0, min(1.0, candidate.score))
        return 0.75 * pair_score + 0.25 * prior

    def score_pair(
        self,
        internal_vector: torch.Tensor,
        candidate_vector: torch.Tensor,
        internal_summary: str,
        candidate: KnowledgeCandidate,
    ) -> tuple[float, torch.Tensor, dict[str, float]]:
        """Score relevance from [q; k; q*k; |q-k|] plus lexical priors."""

        if candidate_vector.numel() == 0:
            candidate_vector = hashed_vector(candidate.text, dim=self.hidden_dim).to(internal_vector.device)
        else:
            candidate_vector = candidate_vector.to(internal_vector.device)
        q = torch.nn.functional.normalize(internal_vector.float(), dim=0)
        k = torch.nn.functional.normalize(candidate_vector.float(), dim=0)
        lexical = self.cross_encoder.score(internal_summary, candidate.text)
        retriever_prior = max(0.0, min(1.0, candidate.score))
        cosine = float(torch.dot(q, k).clamp(-1.0, 1.0).detach())
        feature = torch.cat(
            [
                q,
                k,
                q * k,
                torch.abs(q - k),
                torch.tensor([lexical, retriever_prior, cosine], dtype=q.dtype, device=q.device),
            ],
            dim=0,
        )
        neural = float(torch.sigmoid(self.feature_mlp(feature)).squeeze().detach())
        score = max(0.0, min(1.0, 0.4 * neural + 0.25 * lexical + 0.2 * retriever_prior + 0.15 * max(0.0, cosine)))
        return score, feature.detach(), {
            "neural_relevance": neural,
            "lexical_relevance": lexical,
            "retriever_prior": retriever_prior,
            "embedding_cosine": cosine,
        }


# =============================================================================
# Support / contradiction verification
# =============================================================================

class SupportContradictionVerifier:
    """Classify whether a knowledge item supports the internal hypothesis."""

    def __init__(self) -> None:
        self.cross_encoder = CrossEncoderAdapter()

    def verify(self, internal_summary: str, candidate_text: str) -> tuple[str, float]:
        """Return support label and confidence-like score."""

        label, score = self.cross_encoder.classify_support(internal_summary, candidate_text)
        if label == "irrelevant":
            label = "insufficient"
        return label, score

    def verify_claims(self, claims: dict[str, str], candidate_text: str) -> dict[str, dict[str, float | str]]:
        """Verify target/intent/tactic claims independently."""

        results: dict[str, dict[str, float | str]] = {}
        for claim_type, claim in claims.items():
            if not claim.strip():
                results[claim_type] = {"label": "insufficient", "score": 0.0}
                continue
            label, score = self.verify(claim, candidate_text)
            results[claim_type] = {"label": label, "score": float(score)}
        return results


# =============================================================================
# Validity and temporal checks
# =============================================================================

class CredibilityTemporalValidator:
    """Apply simple source and compatibility checks."""

    def validate(self, candidate: KnowledgeCandidate, internal_text: str = "") -> tuple[float, dict[str, float]]:
        """Return validity score and component scores."""

        source = candidate.source.lower()
        path = str(candidate.metadata.get("path", "")).lower()
        if source == "fallback":
            credibility = 0.45
        elif "wikipedia" in source or "wiki" in source or "wiki" in path:
            credibility = 0.78
        elif "rag" in source or "rag" in path:
            credibility = 0.72
        elif source == "template_generator":
            credibility = 0.5
        else:
            credibility = 0.62
        temporal = _temporal_score(candidate)
        internal_lang = language_hint(internal_text)
        candidate_lang = language_hint(candidate.text)
        language_match = 1.0 if internal_lang == "unknown" or candidate_lang == "unknown" or internal_lang == candidate_lang else 0.55
        cultural = 0.68 if any(word in candidate.text.lower() for word in ["meme", "culture", "context", "reference"]) else 0.55
        cultural = 0.5 * cultural + 0.5 * language_match
        validity = 0.5 * credibility + 0.25 * temporal + 0.25 * cultural
        return validity, {"credibility": credibility, "temporal": temporal, "cultural": cultural, "language_match": language_match}


def _temporal_score(candidate: KnowledgeCandidate) -> float:
    timestamp = candidate.metadata.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp:
        raw = candidate.metadata.get("raw", {})
        timestamp = raw.get("rev_timestamp") if isinstance(raw, dict) else None
    if not isinstance(timestamp, str) or not timestamp:
        return 0.64
    try:
        value = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return 0.6
    now = datetime.now(timezone.utc)
    age_days = max(0, (now - value).days)
    if age_days < 365:
        return 0.8
    if age_days < 365 * 5:
        return 0.68
    return 0.55


# =============================================================================
# Redundancy reduction
# =============================================================================

class RedundancyReducer:
    """Remove near-duplicate knowledge while preserving high scores."""

    def reduce(self, items: list[VerifiedKnowledgeItem], top_k: int = 8, duplicate_threshold: float = 0.82) -> list[VerifiedKnowledgeItem]:
        """Return a diverse top-k subset."""

        selected: list[VerifiedKnowledgeItem] = []
        for item in sorted(items, key=lambda value: value.final_score, reverse=True):
            if any(jaccard_similarity(item.text, chosen.text) >= duplicate_threshold for chosen in selected):
                continue
            selected.append(item)
            if len(selected) >= top_k:
                break
        return selected

    def reduce_with_tokens(
        self,
        items: list[VerifiedKnowledgeItem],
        tokens: dict[str, torch.Tensor],
        top_k: int = 8,
        cluster_threshold: float = 0.78,
    ) -> list[VerifiedKnowledgeItem]:
        """Select high-scoring items while discouraging one-cluster collapse."""

        selected: list[VerifiedKnowledgeItem] = []
        cluster_counts: list[int] = []
        centroids: list[torch.Tensor] = []
        for item in sorted(items, key=lambda value: value.final_score, reverse=True):
            token = tokens.get(item.knowledge_id, hashed_vector(item.text))
            token = F.normalize(token.float(), dim=0)
            cluster_idx = _assign_cluster(token, centroids, cluster_threshold)
            diversity_penalty = 0.08 * cluster_counts[cluster_idx] if cluster_idx is not None else 0.0
            adjusted_score = item.final_score - diversity_penalty
            if adjusted_score <= 0 and len(selected) >= max(1, top_k // 2):
                continue
            if cluster_idx is None:
                centroids.append(token)
                cluster_counts.append(1)
            else:
                cluster_counts[cluster_idx] += 1
                centroids[cluster_idx] = F.normalize(centroids[cluster_idx] * 0.7 + token * 0.3, dim=0)
            item.metadata.setdefault("redundancy", {})
            item.metadata["redundancy"].update({"cluster": cluster_idx if cluster_idx is not None else len(centroids) - 1, "adjusted_score": adjusted_score})
            selected.append(item)
            if len(selected) >= top_k:
                break
        return selected


def _assign_cluster(token: torch.Tensor, centroids: list[torch.Tensor], threshold: float) -> int | None:
    if not centroids:
        return None
    similarities = [float(torch.dot(token.detach(), centroid.detach())) for centroid in centroids]
    best_idx, best_score = max(enumerate(similarities), key=lambda pair: pair[1])
    return best_idx if best_score >= threshold else None


# =============================================================================
# Verification trace and provenance helpers
# =============================================================================



# =============================================================================
# KnowledgeRelevanceFilterVerifier
# =============================================================================

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


__all__ = [
    "VerifiedKnowledgeItem",
    "StageCInput",
    "StageCMetadata",
    "StageCOutput",
    "EvidenceAwareRelevanceScorer",
    "SupportContradictionVerifier",
    "CredibilityTemporalValidator",
    "RedundancyReducer",
    "KnowledgeRelevanceFilterVerifier",
    "SUPPORT_MATRIX_COLUMNS",
]
