"""Evidence attribution for final predictions."""

from __future__ import annotations

from typing import Any

import torch

from module.stage_a.schemas import StageAOutput
from module.stage_c.schemas import StageCOutput
from module.stage_d.schemas import StageDOutput


class EvidenceAttributionLayer:
    """Select supporting internal and external evidence items."""

    def select(
        self,
        stage_a: StageAOutput,
        stage_c: StageCOutput,
        stage_d: StageDOutput,
        top_k: int = 3,
        task: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Return selected evidence dictionaries."""

        token_gates = stage_d.gates.get("token_level")
        if isinstance(token_gates, torch.Tensor):
            token_gates = token_gates.detach()
        else:
            token_gates = torch.ones(len(stage_a.evidence_items))
        head_level = stage_d.gates.get("head_level")
        task_idx = {"target": 0, "intent": 1, "tactic": 2}.get(task or "", None)
        internal_ranked = []
        for item in stage_a.evidence_items:
            gate_score = float(token_gates[item.token_index]) if item.token_index < token_gates.numel() else item.score
            if isinstance(head_level, torch.Tensor) and task_idx is not None and item.token_index < head_level.size(1):
                gate_score = 0.7 * gate_score + 0.3 * float(head_level.detach()[task_idx, item.token_index])
            type_bonus = 1.1 if item.evidence_type in {"text_span", "cross_modal_incongruity", "local_symbol"} else 1.0
            internal_ranked.append((gate_score * max(item.score, 0.05) * type_bonus, item))
        internal_ranked.sort(key=lambda pair: pair[0], reverse=True)
        internal = [
            {
                "evidence_id": item.evidence_id,
                "type": item.evidence_type,
                "text": item.text,
                "score": float(score),
                "metadata": item.metadata,
            }
            for score, item in internal_ranked[:top_k]
        ]

        attention = stage_d.cross_attention_weights.detach() if isinstance(stage_d.cross_attention_weights, torch.Tensor) else torch.zeros(0, 0)
        external_ranked = []
        for idx, item in enumerate(stage_c.verified_items):
            attention_score = float(attention[:, idx].mean()) if attention.numel() and idx < attention.size(1) else 0.0
            support_bonus = 0.08 if item.support_label == "support" else -0.02 if item.support_label == "contradict" else 0.0
            external_ranked.append((item.final_score + 0.25 * attention_score + support_bonus, attention_score, item))
        external_ranked.sort(key=lambda row: row[0], reverse=True)
        external = [
            {
                "knowledge_id": item.knowledge_id,
                "text": item.text,
                "source": item.source,
                "score": float(score),
                "final_score": item.final_score,
                "attention_score": float(attention_score),
                "support_label": item.support_label,
                "metadata": item.metadata,
            }
            for score, attention_score, item in external_ranked[:top_k]
        ]
        return {"internal": internal, "external": external}
