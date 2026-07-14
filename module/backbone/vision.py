"""Vision backbone adapters for Stage A and baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from utils.image_utils import fallback_image_embedding, image_size
from utils.logging_utils import setup_logger
from utils.tensor_utils import hashed_vector
from utils.text_utils import capitalized_spans, keyword_candidates, rhetorical_cues


# =============================================================================
# CLIP image wrapper
# =============================================================================

logger = setup_logger(__name__)


class CLIPWrapper(nn.Module):
    """Expose a stable image encoding API while optional CLIP deps are absent."""

    def __init__(
        self,
        hidden_dim: int = 256,
        prefer_pretrained: bool = False,
        model_name: str = "ViT-B-32",
        device: str = "cpu",
        pretrained_tag: str | None = None,
        checkpoint_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = True,
        allow_download: bool = False,
        asset_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.device_name = device
        self.model_name = model_name
        self.prefer_pretrained = prefer_pretrained
        self.pretrained_tag = pretrained_tag
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.local_files_only = local_files_only
        self.allow_download = allow_download
        self.asset_mode = asset_mode or ("local_checkpoint" if self.checkpoint_path else "pretrained_tag" if pretrained_tag else "fallback")
        self.model: Any | None = None
        self.preprocess: Any | None = None
        self.backend = "fallback"
        self._readiness: dict[str, Any] = {
            "requested_backend": "clip",
            "resolved_backend": "fallback",
            "model_name": model_name,
            "prefer_pretrained": prefer_pretrained,
            "pretrained_requested": bool(prefer_pretrained),
            "pretrained_tag": pretrained_tag,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "resolved_path": str(self.checkpoint_path.resolve()) if self.checkpoint_path else None,
            "asset_mode": self.asset_mode,
            "checkpoint_exists": bool(self.checkpoint_path and self.checkpoint_path.exists()),
            "checkpoint_sha256": _sha256_file(self.checkpoint_path),
            "checkpoint_format": None,
            "checkpoint_compatibility_verified": False,
            "checkpoint_model_name": None,
            "checkpoint_parameter_key_count": 0,
            "model_parameter_key_count": 0,
            "matched_parameter_key_count": 0,
            "matched_parameter_numel": 0,
            "model_parameter_numel": 0,
            "matched_parameter_ratio": None,
            "missing_key_count": 0,
            "unexpected_key_count": 0,
            "shape_mismatch_count": 0,
            "compatibility_failure_reason": None,
            "factory_local_load_error": None,
            "weights_loaded": False,
            "weights_source": None,
            "local_files_only": local_files_only,
            "allow_download": allow_download,
            "fallback_used": True,
            "random_initialization_used": False,
            "load_error": None,
        }
        self._projection: nn.Linear | None = None
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)
        if prefer_pretrained:
            self._try_load_clip(model_name, device)

    def _try_load_clip(self, model_name: str, device: str) -> None:
        if self.asset_mode == "local_checkpoint":
            if not self.checkpoint_path:
                self._mark_load_error("asset_mode=local_checkpoint requires checkpoint_path")
                return
            if not self.checkpoint_path.exists():
                self._mark_load_error(f"checkpoint_path does not exist: {self.checkpoint_path}")
                return
            if not self.checkpoint_path.is_file():
                self._mark_load_error(f"checkpoint_path is not a file: {self.checkpoint_path}")
                return
            if self.checkpoint_path.stat().st_size <= 0:
                self._mark_load_error(f"checkpoint_path is empty: {self.checkpoint_path}")
                return
        if self.local_files_only and not self.allow_download and not self.checkpoint_path:
            self._mark_load_error("pretrained vision requested, but no local checkpoint_path was configured")
            return
        try:
            import open_clip  # type: ignore
        except Exception as exc:
            self._mark_load_error(f"open_clip: {_short_error(exc)}")
            logger.info("open_clip unavailable, trying clip package: %s", exc)
            open_clip = None  # type: ignore

        if open_clip is not None and self.asset_mode == "local_checkpoint" and self.checkpoint_path:
            if self._try_open_clip_factory_local(open_clip, model_name, device):
                return
            if self._try_open_clip_manual_checkpoint(open_clip, model_name, device):
                return
            return

        if open_clip is not None:
            try:
                pretrained = self.pretrained_tag if (self.allow_download and self.pretrained_tag and self.asset_mode != "local_checkpoint") else None
                model, _, preprocess = _open_clip_create_model_and_transforms(
                    open_clip,
                    model_name,
                    pretrained=pretrained,
                    cache_dir=str(self.cache_dir) if self.cache_dir else None,
                )
                weights_loaded = bool(pretrained)
                weights_source = f"open_clip:{pretrained}" if pretrained else None
                self.model = model.to(device).eval()
                self.preprocess = preprocess
                self.backend = "open_clip"
                self._readiness.update(
                    {
                        "resolved_backend": "open_clip",
                        "weights_loaded": weights_loaded,
                        "weights_source": weights_source,
                        "fallback_used": False,
                        "random_initialization_used": not weights_loaded,
                        "load_error": None if weights_loaded else "open_clip model initialized without pretrained weights",
                    }
                )
                logger.info("Using open_clip backend: %s", model_name)
                return
            except Exception as exc:
                self._mark_load_error(f"open_clip: {_short_error(exc)}")
                logger.info("open_clip unavailable, trying clip package: %s", exc)
        if not self.allow_download:
            logger.info("clip package loading skipped because allow_download=false")
            return
        try:
            import clip  # type: ignore

            model, preprocess = clip.load(model_name, device=device)
            self.model = model.eval()
            self.preprocess = preprocess
            self.backend = "clip"
            self._readiness.update(
                {
                    "resolved_backend": "clip",
                    "weights_loaded": True,
                    "weights_source": f"clip:{model_name}",
                    "fallback_used": False,
                    "random_initialization_used": False,
                    "load_error": None,
                }
            )
            logger.info("Using clip backend: %s", model_name)
        except Exception as exc:
            self._mark_load_error(f"clip: {_short_error(exc)}")
            logger.info("CLIP unavailable; using fallback image encoder: %s", exc)

    def _try_open_clip_factory_local(self, open_clip: Any, model_name: str, device: str) -> bool:
        assert self.checkpoint_path is not None
        try:
            model, _, preprocess = _open_clip_create_model_and_transforms(
                open_clip,
                model_name,
                pretrained=str(self.checkpoint_path),
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
            )
            self.model = model.to(device).eval()
            self.preprocess = preprocess
            self.backend = "open_clip"
            model_state = self.model.state_dict() if hasattr(self.model, "state_dict") else {}
            model_numel = _state_numel(model_state)
            self._readiness.update(
                {
                    "resolved_backend": "open_clip",
                    "weights_loaded": True,
                    "weights_source": "local_checkpoint",
                    "checkpoint_format": "open_clip_factory_local_path",
                    "checkpoint_compatibility_verified": True,
                    "checkpoint_model_name": model_name,
                    "model_parameter_key_count": len(model_state),
                    "model_parameter_numel": model_numel,
                    "matched_parameter_key_count": len(model_state),
                    "matched_parameter_numel": model_numel,
                    "matched_parameter_ratio": 1.0,
                    "missing_key_count": 0,
                    "unexpected_key_count": 0,
                    "shape_mismatch_count": 0,
                    "compatibility_failure_reason": None,
                    "resolved_path": str(self.checkpoint_path.resolve()),
                    "checkpoint_sha256": _sha256_file(self.checkpoint_path),
                    "fallback_used": False,
                    "random_initialization_used": False,
                    "load_error": None,
                }
            )
            logger.info("Using open_clip local checkpoint via factory: %s", self.checkpoint_path)
            return True
        except Exception as exc:
            message = _short_error(exc)
            self._readiness["factory_local_load_error"] = message
            self._readiness["load_error"] = f"factory_local_load_failed: {message}"
            logger.info("open_clip factory local checkpoint load failed: %s", exc)
            return False

    def _try_open_clip_manual_checkpoint(self, open_clip: Any, model_name: str, device: str) -> bool:
        assert self.checkpoint_path is not None
        try:
            model, _, preprocess = _open_clip_create_model_and_transforms(
                open_clip,
                model_name,
                pretrained=None,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
            )
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            state_dict, checkpoint_model_name = _extract_checkpoint_state_dict(checkpoint)
            model_state = model.state_dict()
            diagnostics = _state_dict_compatibility_diagnostics(state_dict, model_state)
            diagnostics["checkpoint_model_name"] = checkpoint_model_name
            self._readiness.update(diagnostics)
            self._readiness.update(
                {
                    "resolved_backend": "open_clip",
                    "checkpoint_format": "manual_state_dict_validated",
                    "resolved_path": str(self.checkpoint_path.resolve()),
                    "checkpoint_sha256": _sha256_file(self.checkpoint_path),
                }
            )
            failure = _manual_checkpoint_failure(diagnostics)
            if failure is not None:
                self._mark_compatibility_failure(failure)
                return False
            normalized = _normalize_checkpoint_keys(state_dict)
            loadable = {key: normalized[key] for key in model_state if key in normalized}
            strict_load = len(loadable) == len(model_state) and diagnostics["unexpected_key_count"] == 0
            model.load_state_dict(loadable, strict=strict_load)
            self.model = model.to(device).eval()
            self.preprocess = preprocess
            self.backend = "open_clip"
            self._readiness.update(
                {
                    "weights_loaded": True,
                    "weights_source": "local_checkpoint",
                    "checkpoint_compatibility_verified": True,
                    "fallback_used": False,
                    "random_initialization_used": False,
                    "compatibility_failure_reason": None,
                    "load_error": None,
                }
            )
            logger.info("Using open_clip local checkpoint via validated manual load: %s", self.checkpoint_path)
            return True
        except Exception as exc:
            self._mark_compatibility_failure(f"manual_checkpoint_load_failed: {_short_error(exc)}")
            return False

    @torch.no_grad()
    def encode_image(self, image_path: str | Path | None) -> torch.Tensor:
        """Encode a whole image to the configured hidden dimension."""

        target_device = self._target_device()
        if self.model is not None and self.preprocess is not None and image_path is not None:
            try:
                from PIL import Image

                image = Image.open(image_path).convert("RGB")
                tensor = self.preprocess(image).unsqueeze(0).to(self.device_name)
                feature = self.model.encode_image(tensor).float().squeeze(0).to(target_device)
                return self._project_feature(feature)
            except Exception as exc:
                logger.warning("CLIP image encoding failed for %s; falling back: %s", image_path, exc)
        return fallback_image_embedding(image_path, dim=self.hidden_dim).to(target_device)

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text in the same CLIP space used for images.

        Stage A keeps its dedicated text encoder. This additive method exists
        for the supervised shared-OpenCLIP baseline and retains deterministic
        offline fallback behavior.
        """

        target_device = self._target_device()
        if self.model is not None:
            try:
                if self.backend == "open_clip":
                    import open_clip  # type: ignore

                    tokenizer = open_clip.get_tokenizer(self.model_name)
                    tokens = tokenizer([text or ""]).to(self.device_name)
                elif self.backend == "clip":
                    import clip  # type: ignore

                    tokens = clip.tokenize([text or ""], truncate=True).to(self.device_name)
                else:
                    raise RuntimeError(f"unsupported CLIP text backend: {self.backend}")
                feature = self.model.encode_text(tokens).float().squeeze(0).to(target_device)
                return self._project_feature(feature)
            except Exception as exc:
                logger.warning("CLIP text encoding failed; falling back: %s", exc)
        return F.normalize(hashed_vector(f"clip-text:{text or ''}", dim=self.hidden_dim), dim=0).to(target_device)

    @torch.no_grad()
    def encode_rois(self, image_path: str | Path | None, boxes: list[tuple[float, float, float, float]]) -> torch.Tensor:
        """Encode ROI boxes; fallback appends box coordinates to the image identity."""

        target_device = self._target_device()
        if not boxes:
            return torch.zeros(0, self.hidden_dim, device=target_device)
        vectors = []
        for box in boxes:
            box_text = " ".join(f"{value:.3f}" for value in box)
            base = self.encode_image(image_path)
            roi_hint = hashed_vector(f"roi:{image_path}:{box_text}", dim=self.hidden_dim).to(base.device)
            vectors.append(F.normalize(0.7 * base + 0.3 * roi_hint, dim=0))
        return torch.stack(vectors, dim=0)

    def _project_feature(self, feature: torch.Tensor) -> torch.Tensor:
        feature = feature.flatten().float()
        if feature.numel() == self.hidden_dim:
            return F.normalize(feature, dim=0)
        if self._projection is None or self._projection.in_features != feature.numel():
            self._projection = nn.Linear(feature.numel(), self.hidden_dim).to(feature.device)
        else:
            self._projection = self._projection.to(feature.device)
        projected = self._projection(feature.unsqueeze(0)).squeeze(0)
        return F.normalize(projected, dim=0)

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
        """Return serializable backbone readiness and provenance state."""

        state = dict(self._readiness)
        state["resolved_backend"] = self.backend
        state["checkpoint_exists"] = bool(self.checkpoint_path and self.checkpoint_path.exists())
        state["checkpoint_sha256"] = _sha256_file(self.checkpoint_path)
        state["resolved_path"] = str(self.checkpoint_path.resolve()) if self.checkpoint_path else None
        state["fallback_used"] = bool(state.get("fallback_used"))
        if state.get("weights_loaded") and not state.get("checkpoint_compatibility_verified") and self.asset_mode == "local_checkpoint":
            state["weights_loaded"] = False
        return state

    def _mark_load_error(self, message: str, *, fallback_used: bool = True) -> None:
        self._readiness.update(
            {
                "resolved_backend": self.backend,
                "weights_loaded": False,
                "weights_source": None,
                "checkpoint_compatibility_verified": False,
                "fallback_used": fallback_used,
                "random_initialization_used": False,
                "load_error": message[:500],
            }
        )

    def _mark_compatibility_failure(self, reason: str) -> None:
        self._readiness.update(
            {
                "resolved_backend": "open_clip",
                "weights_loaded": False,
                "weights_source": None,
                "checkpoint_compatibility_verified": False,
                "fallback_used": False,
                "random_initialization_used": False,
                "compatibility_failure_reason": reason[:500],
                "load_error": reason[:500],
            }
        )


# =============================================================================
# Detector adapter
# =============================================================================

@dataclass
class Detection:
    """A local region proposal or symbolic cue."""

    box: tuple[float, float, float, float]
    label: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class DetectorAdapter:
    """Phase-1 detector adapter with an explicit fallback mode."""

    def __init__(self, mode: str = "heuristic", max_rois: int = 3) -> None:
        self.mode = mode
        self.max_rois = max_rois

    def detect(self, image_path: str | Path | None, ocr_text: str = "") -> list[Detection]:
        """Return detections; heuristic mode creates broad pseudo-regions."""

        if self.mode == "empty":
            return []
        size = image_size(image_path)
        entities = capitalized_spans(ocr_text, limit=self.max_rois)
        cues = list(rhetorical_cues(ocr_text).keys())
        labels = [*entities, *cues, *keyword_candidates(ocr_text, limit=self.max_rois)] or ["whole_image"]
        detections: list[Detection] = []
        if size is None:
            for idx, label in enumerate(labels[: self.max_rois]):
                detections.append(
                    Detection(
                        box=(0.0, 0.0, 1.0, 1.0),
                        label=f"pseudo_{label}",
                        score=max(0.25, 0.55 - idx * 0.08),
                        metadata={"fallback": True},
                    )
                )
            return detections

        width, height = size
        templates = [
            (0.0, 0.0, width, height),
            (0.0, 0.0, width, height * 0.45),
            (0.0, height * 0.55, width, height),
            (0.0, height * 0.25, width * 0.5, height * 0.75),
            (width * 0.5, height * 0.25, width, height * 0.75),
        ]
        for idx, label in enumerate(labels[: self.max_rois]):
            box = templates[idx % len(templates)]
            detections.append(
                Detection(
                    box=tuple(float(v) for v in box),
                    label=f"region_{label}",
                    score=max(0.3, 0.7 - idx * 0.1),
                    metadata={"image_size": [width, height], "fallback": True},
                )
            )
        return detections


__all__ = ["CLIPWrapper", "Detection", "DetectorAdapter"]


def _short_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc)[:300]}"


def _open_clip_create_model_and_transforms(open_clip: Any, model_name: str, *, pretrained: str | None, cache_dir: str | None):
    kwargs = {"pretrained": pretrained}
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    try:
        return open_clip.create_model_and_transforms(model_name, **kwargs)
    except TypeError:
        kwargs.pop("cache_dir", None)
        return open_clip.create_model_and_transforms(model_name, **kwargs)


def _extract_checkpoint_state_dict(checkpoint: Any) -> tuple[dict[str, torch.Tensor], str | None]:
    checkpoint_model_name = None
    state = checkpoint
    if isinstance(checkpoint, dict):
        checkpoint_model_name = checkpoint.get("model_name") or checkpoint.get("arch") or checkpoint.get("model")
        for key in ["state_dict", "model_state_dict", "model", "module"]:
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict) and candidate and all(isinstance(value, torch.Tensor) for value in candidate.values()):
                state = candidate
                break
        else:
            if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
                state = checkpoint
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain a tensor state_dict")
    tensor_state = {str(key): value for key, value in state.items() if isinstance(value, torch.Tensor)}
    if not tensor_state:
        raise ValueError("checkpoint state_dict has no tensor parameters")
    return tensor_state, str(checkpoint_model_name) if checkpoint_model_name else None


def _normalize_checkpoint_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    prefixes = ["module.", "model."]
    for key, value in state_dict.items():
        current = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if current.startswith(prefix):
                    current = current[len(prefix) :]
                    changed = True
        normalized[current] = value
    return normalized


def _state_dict_compatibility_diagnostics(
    checkpoint_state: dict[str, torch.Tensor],
    model_state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    normalized = _normalize_checkpoint_keys(checkpoint_state)
    checkpoint_keys = set(normalized)
    model_keys = set(model_state)
    matched_keys = []
    shape_mismatches = []
    matched_numel = 0
    for key in sorted(checkpoint_keys & model_keys):
        checkpoint_tensor = normalized[key]
        model_tensor = model_state[key]
        if tuple(checkpoint_tensor.shape) == tuple(model_tensor.shape):
            matched_keys.append(key)
            matched_numel += int(model_tensor.numel())
        else:
            shape_mismatches.append(key)
    model_numel = _state_numel(model_state)
    return {
        "checkpoint_parameter_key_count": len(checkpoint_keys),
        "model_parameter_key_count": len(model_keys),
        "matched_parameter_key_count": len(matched_keys),
        "matched_parameter_numel": matched_numel,
        "model_parameter_numel": model_numel,
        "matched_parameter_ratio": (matched_numel / model_numel) if model_numel else 1.0,
        "missing_key_count": len(model_keys - set(matched_keys)),
        "unexpected_key_count": len(checkpoint_keys - model_keys),
        "shape_mismatch_count": len(shape_mismatches),
        "missing_key_samples": sorted(model_keys - set(matched_keys))[:5],
        "unexpected_key_samples": sorted(checkpoint_keys - model_keys)[:5],
        "shape_mismatch_samples": shape_mismatches[:5],
    }


def _manual_checkpoint_failure(diagnostics: dict[str, Any]) -> str | None:
    if int(diagnostics.get("shape_mismatch_count") or 0) > 0:
        return "checkpoint_shape_mismatch"
    if int(diagnostics.get("matched_parameter_key_count") or 0) == 0:
        return "checkpoint_key_mismatch_zero_matched"
    ratio = diagnostics.get("matched_parameter_ratio")
    if ratio is None or float(ratio) < 0.99:
        return f"checkpoint_key_mismatch_low_coverage:{ratio}"
    return None


def _state_numel(state: dict[str, torch.Tensor]) -> int:
    return sum(int(tensor.numel()) for tensor in state.values() if isinstance(tensor, torch.Tensor))


def _sha256_file(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    try:
        from experiments.run_manifest import sha256_file

        return sha256_file(path)
    except Exception:
        return None
