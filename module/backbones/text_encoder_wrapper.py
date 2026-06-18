"""Text encoder wrapper with HuggingFace and hashing fallbacks."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from utils.logging_utils import setup_logger
from utils.tensor_utils import hashed_vector
from utils.text_utils import tokenize


logger = setup_logger(__name__)


class TextEncoderWrapper(nn.Module):
    """Encode OCR text into a global vector and token-level embeddings."""

    def __init__(
        self,
        hidden_dim: int = 256,
        prefer_transformers: bool = False,
        model_name: str = "microsoft/deberta-v3-base",
        max_tokens: int = 64,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_tokens = max_tokens
        self.device_name = device
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self.backend = "hashing"
        self._projection: nn.Linear | None = None
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)
        if prefer_transformers:
            self._try_load_transformer(model_name, device)

    def _try_load_transformer(self, model_name: str, device: str) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer  # type: ignore

            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModel.from_pretrained(model_name, local_files_only=True).to(device).eval()
            self.backend = "transformers"
            logger.info("Using local HuggingFace text encoder: %s", model_name)
        except Exception as exc:
            logger.info("Transformers encoder unavailable; using hashing fallback: %s", exc)

    @torch.no_grad()
    def encode(self, text: str) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        """Return global embedding, token embeddings, and token strings."""

        target_device = self._target_device()
        if self.model is not None and self.tokenizer is not None:
            try:
                encoded = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_tokens,
                    return_tensors="pt",
                ).to(self.device_name)
                output = self.model(**encoded)
                token_features = output.last_hidden_state.squeeze(0).float().to(target_device)
                token_features = self._project_matrix(token_features)
                global_feature = F.normalize(token_features.mean(dim=0), dim=0)
                tokens = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"].squeeze(0).cpu().tolist())
                return global_feature, token_features, tokens
            except Exception as exc:
                logger.warning("Transformer text encoding failed; using hashing fallback: %s", exc)

        tokens = tokenize(text)[: self.max_tokens]
        if not tokens:
            tokens = ["<empty>"]
        token_features = torch.stack([hashed_vector(token, dim=self.hidden_dim) for token in tokens], dim=0).to(target_device)
        global_feature = F.normalize(token_features.mean(dim=0), dim=0)
        return global_feature, token_features, tokens

    def _project_matrix(self, matrix: torch.Tensor) -> torch.Tensor:
        if matrix.size(-1) == self.hidden_dim:
            return F.normalize(matrix, dim=-1)
        if self._projection is None or self._projection.in_features != matrix.size(-1):
            self._projection = nn.Linear(matrix.size(-1), self.hidden_dim).to(matrix.device)
        else:
            self._projection = self._projection.to(matrix.device)
        return F.normalize(self._projection(matrix), dim=-1)

    def _target_device(self) -> torch.device:
        for param in self.parameters():
            return param.device
        for buffer in self.buffers():
            return buffer.device
        try:
            return torch.device(self.device_name)
        except (TypeError, RuntimeError):
            return torch.device("cpu")
