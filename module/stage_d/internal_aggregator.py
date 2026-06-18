"""Internal evidence aggregation for Stage D."""

from __future__ import annotations

import torch
from torch import nn

from module.stage_a.schemas import StageAOutput
from utils.tensor_utils import stable_hash


class InternalEvidenceAggregator(nn.Module):
    """Aggregate internal evidence with type and dataset embeddings."""

    def __init__(self, hidden_dim: int = 256, num_heads: int = 4, num_layers: int = 2, dataset_buckets: int = 32) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.type_to_id = {
            "global_visual": 0,
            "global_text": 1,
            "local_symbol": 2,
            "cross_modal_incongruity": 3,
            "text_span": 4,
            "visual_patch": 5,
        }
        self.type_embedding = nn.Embedding(8, hidden_dim)
        self.dataset_embedding = nn.Embedding(dataset_buckets, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dataset_buckets = dataset_buckets

    def forward(self, stage_a: StageAOutput) -> tuple[torch.Tensor, torch.Tensor]:
        """Return contextualized internal memory and a summary vector."""

        device = self.type_embedding.weight.device
        tokens = stage_a.internal_tokens.to(device=device, dtype=self.type_embedding.weight.dtype)
        if tokens.numel() == 0:
            tokens = torch.zeros(1, self.hidden_dim, device=device)
        type_ids = []
        for item in stage_a.evidence_items[: tokens.size(0)]:
            type_ids.append(self.type_to_id.get(item.evidence_type, 7))
        while len(type_ids) < tokens.size(0):
            type_ids.append(7)
        type_tensor = torch.tensor(type_ids, dtype=torch.long, device=tokens.device)
        dataset_id = stable_hash(stage_a.dataset_name, self.dataset_buckets)
        dataset_tensor = torch.full((tokens.size(0),), dataset_id, dtype=torch.long, device=tokens.device)
        enriched = tokens + self.type_embedding(type_tensor) + self.dataset_embedding(dataset_tensor)
        memory = self.encoder(enriched.unsqueeze(0)).squeeze(0)
        memory = self.norm(memory)
        summary = memory.mean(dim=0)
        return memory, summary
