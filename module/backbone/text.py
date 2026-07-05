"""Text backbone adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from utils.logging_utils import setup_logger
from utils.tensor_utils import hashed_vector
from utils.text_utils import tokenize


# =============================================================================
# Text encoder wrapper
# =============================================================================

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
        checkpoint_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        tokenizer_use_fast: bool = False,
        tokenizer_backend_policy: str | None = None,
        local_files_only: bool = True,
        allow_download: bool = False,
        asset_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_tokens = max_tokens
        self.device_name = device
        self.model_name = model_name
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.local_files_only = local_files_only
        self.allow_download = allow_download
        self.asset_mode = asset_mode or ("local_directory" if self.checkpoint_path else "model_name")
        self.tokenizer_use_fast = bool(tokenizer_use_fast)
        local_deberta = _is_local_deberta_v3(self.model_name, self.checkpoint_path, self.asset_mode)
        self.tokenizer_backend_policy = tokenizer_backend_policy or ("sentencepiece_slow" if local_deberta else "auto")
        self.sentencepiece_required = bool(local_deberta or self.tokenizer_backend_policy == "sentencepiece_slow")
        self.sentencepiece_available = _sentencepiece_available()
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self.backend = "hashing"
        self._readiness: dict[str, Any] = {
            "requested_backend": "transformers",
            "resolved_backend": "hashing",
            "model_name": model_name,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "resolved_path": str(self.checkpoint_path.resolve()) if self.checkpoint_path else None,
            "asset_mode": self.asset_mode,
            "checkpoint_exists": bool(self.checkpoint_path and self.checkpoint_path.exists()),
            "checkpoint_sha256": _path_sha256(self.checkpoint_path),
            "weights_loaded": False,
            "weights_source": None,
            "local_files_only": local_files_only,
            "allow_download": allow_download,
            "tokenizer_use_fast": self.tokenizer_use_fast,
            "tokenizer_backend_policy": self.tokenizer_backend_policy,
            "tokenizer_class": None,
            "tokenizer_loaded": False,
            "sentencepiece_required": self.sentencepiece_required,
            "sentencepiece_available": self.sentencepiece_available,
            "fallback_used": True,
            "load_error": None,
        }
        self._projection: nn.Linear | None = None
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)
        if prefer_transformers:
            self._try_load_transformer(model_name, device)

    def _try_load_transformer(self, model_name: str, device: str) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer  # type: ignore

            source = str(self.checkpoint_path) if self.checkpoint_path else model_name
            if self.asset_mode == "local_directory":
                if not self.checkpoint_path:
                    self._mark_load_error("asset_mode=local_directory requires checkpoint_path")
                    return
                if not self.checkpoint_path.exists():
                    self._mark_load_error(f"checkpoint_path does not exist: {self.checkpoint_path}")
                    return
                if not self.checkpoint_path.is_dir():
                    self._mark_load_error(f"checkpoint_path is not a directory: {self.checkpoint_path}")
                    return
                local_only = True
            else:
                if self.checkpoint_path and not self.checkpoint_path.exists():
                    self._mark_load_error(f"checkpoint_path does not exist: {self.checkpoint_path}")
                    return
                local_only = True if self.checkpoint_path else self.local_files_only or not self.allow_download
            kwargs = {
                "local_files_only": local_only,
            }
            if self.cache_dir:
                kwargs["cache_dir"] = str(self.cache_dir)
            tokenizer_kwargs = dict(kwargs)
            tokenizer_kwargs["use_fast"] = self.tokenizer_use_fast
            self.tokenizer = AutoTokenizer.from_pretrained(source, **tokenizer_kwargs)
            self._readiness.update(
                {
                    "tokenizer_loaded": True,
                    "tokenizer_class": type(self.tokenizer).__name__,
                    "sentencepiece_available": _sentencepiece_available(),
                }
            )
            self.model = AutoModel.from_pretrained(source, **kwargs).to(device).eval()
            self.backend = "transformers"
            self._readiness.update(
                {
                    "resolved_backend": "transformers",
                    "weights_loaded": self.tokenizer is not None and self.model is not None,
                    "weights_source": "local_directory" if self.asset_mode == "local_directory" else source,
                    "resolved_path": str(self.checkpoint_path.resolve()) if self.checkpoint_path else None,
                    "checkpoint_sha256": _path_sha256(self.checkpoint_path),
                    "fallback_used": False,
                    "load_error": None,
                }
            )
            logger.info("Using local HuggingFace text encoder: %s", model_name)
        except Exception as exc:
            self._mark_load_error(_short_error(exc))
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

    def readiness_state(self) -> dict[str, Any]:
        """Return serializable text-backbone readiness and provenance state."""

        state = dict(self._readiness)
        self.sentencepiece_available = _sentencepiece_available()
        state["resolved_backend"] = self.backend
        state["checkpoint_exists"] = bool(self.checkpoint_path and self.checkpoint_path.exists())
        state["resolved_path"] = str(self.checkpoint_path.resolve()) if self.checkpoint_path else None
        state["checkpoint_sha256"] = _path_sha256(self.checkpoint_path)
        state["tokenizer_use_fast"] = self.tokenizer_use_fast
        state["tokenizer_backend_policy"] = self.tokenizer_backend_policy
        state["tokenizer_loaded"] = self.tokenizer is not None and bool(state.get("tokenizer_loaded"))
        state["tokenizer_class"] = type(self.tokenizer).__name__ if self.tokenizer is not None else state.get("tokenizer_class")
        state["sentencepiece_required"] = self.sentencepiece_required
        state["sentencepiece_available"] = self.sentencepiece_available
        state["weights_loaded"] = bool(self.model is not None and self.tokenizer is not None and state.get("weights_loaded"))
        state["fallback_used"] = self.model is None or self.tokenizer is None or self.backend == "hashing" or bool(state.get("fallback_used"))
        return state

    def _mark_load_error(self, message: str) -> None:
        self._readiness.update(
            {
                "resolved_backend": self.backend,
                "weights_loaded": False,
                "weights_source": None,
                "tokenizer_loaded": self.tokenizer is not None,
                "tokenizer_class": type(self.tokenizer).__name__ if self.tokenizer is not None else None,
                "sentencepiece_available": _sentencepiece_available(),
                "fallback_used": True,
                "load_error": message[:500],
            }
        )


__all__ = ["TextEncoderWrapper"]


def _short_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc)[:300]}"


def _is_local_deberta_v3(model_name: str, checkpoint_path: Path | None, asset_mode: str | None) -> bool:
    if asset_mode != "local_directory":
        return False
    marker = f"{model_name} {checkpoint_path or ''}".lower().replace("_", "-")
    return "deberta-v3" in marker


def _sentencepiece_available() -> bool:
    try:
        import sentencepiece  # noqa: F401

        return True
    except Exception:
        return False


def _path_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    try:
        from experiments.pretrained_assets import _directory_sha256  # type: ignore
        from experiments.run_manifest import sha256_file

        return _directory_sha256(path) if path.is_dir() else sha256_file(path)
    except Exception:
        return None
