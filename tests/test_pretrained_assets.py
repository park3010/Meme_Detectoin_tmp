from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from experiments.pretrained_assets import (
    build_asset_provenance,
    inspect_text_asset,
    inspect_vision_asset,
    verify_pretrained_assets,
    write_asset_manifest,
)
from experiments.run_manifest import build_run_manifest
from module.backbone.text import TextEncoderWrapper
from module.backbone.vision import CLIPWrapper


def test_vision_asset_inspection_missing_empty_and_sha(tmp_path: Path):
    missing_cfg = _config(tmp_path)
    missing = inspect_vision_asset(missing_cfg)
    assert missing.exists is False
    assert missing.usable is False
    assert any(issue["code"] == "vision_checkpoint_missing" for issue in missing.issues)

    checkpoint = tmp_path / "vision" / "checkpoint.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"")
    empty = inspect_vision_asset(_config(tmp_path, vision_path=checkpoint))
    assert empty.usable is False
    assert any(issue["code"] == "vision_checkpoint_empty" for issue in empty.issues)

    checkpoint.write_bytes(b"not-a-real-checkpoint")
    record = inspect_vision_asset(_config(tmp_path, vision_path=checkpoint))
    assert record.exists is True
    assert record.sha256
    path = write_asset_manifest(record, output_path=tmp_path / "vision_manifest.json")
    json.dumps(json.loads(path.read_text(encoding="utf-8")))


def test_text_asset_inspection_missing_config_only_and_complete(tmp_path: Path):
    missing = inspect_text_asset(_config(tmp_path))
    assert missing.exists is False
    assert missing.usable is False

    text_dir = tmp_path / "text"
    text_dir.mkdir()
    (text_dir / "config.json").write_text("{}", encoding="utf-8")
    config_only = inspect_text_asset(_config(tmp_path, text_path=text_dir))
    assert config_only.usable is False
    assert "model.safetensors|pytorch_model.bin|sharded weights" in config_only.missing_files

    _write_text_snapshot(text_dir, weight_name="model.safetensors")
    complete = inspect_text_asset(_config(tmp_path, text_path=text_dir))
    assert complete.usable is True
    assert complete.sha256

    (text_dir / "model.safetensors").unlink()
    (text_dir / "pytorch_model.bin").write_bytes(b"weights")
    bin_record = inspect_text_asset(_config(tmp_path, text_path=text_dir))
    assert bin_record.usable is True


def test_runtime_verification_fails_with_missing_assets(tmp_path: Path):
    result = verify_pretrained_assets(_config(tmp_path), strict=True)
    assert result.passed is False
    codes = {item["code"] for item in result.errors}
    assert "vision_asset_unusable" in codes
    assert "text_asset_unusable" in codes
    assert "vision_runtime_weights_not_loaded" in codes
    assert "text_runtime_weights_not_loaded" in codes


def test_local_checkpoint_mode_never_requests_remote_download(tmp_path: Path, monkeypatch):
    checkpoint = tmp_path / "vision" / "checkpoint.pt"
    checkpoint.parent.mkdir()
    torch.save({}, checkpoint)
    calls = {}

    fake_open_clip = SimpleNamespace(create_model_and_transforms=lambda model_name, pretrained=None, cache_dir=None: _fake_clip_tuple(calls, pretrained))
    monkeypatch.setitem(sys.modules, "open_clip", fake_open_clip)

    wrapper = CLIPWrapper(
        prefer_pretrained=True,
        checkpoint_path=checkpoint,
        asset_mode="local_checkpoint",
        allow_download=False,
        local_files_only=True,
    )
    state = wrapper.readiness_state()
    assert calls["pretrained"] == str(checkpoint)
    assert state["weights_loaded"] is True
    assert state["checkpoint_compatibility_verified"] is True
    assert state["checkpoint_format"] == "open_clip_factory_local_path"
    assert state["fallback_used"] is False
    assert state["weights_source"] == "local_checkpoint"


def test_mismatched_vision_checkpoint_shape_fails_without_fallback_label(tmp_path: Path, monkeypatch):
    checkpoint = tmp_path / "vision" / "checkpoint.pt"
    checkpoint.parent.mkdir()
    torch.save({"state_dict": {"proj.weight": torch.ones(3, 3), "proj.bias": torch.ones(2)}}, checkpoint)
    monkeypatch.setitem(sys.modules, "open_clip", _manual_open_clip())

    wrapper = CLIPWrapper(prefer_pretrained=True, checkpoint_path=checkpoint, asset_mode="local_checkpoint")
    state = wrapper.readiness_state()

    assert state["weights_loaded"] is False
    assert state["checkpoint_compatibility_verified"] is False
    assert state["shape_mismatch_count"] == 1
    assert state["compatibility_failure_reason"] == "checkpoint_shape_mismatch"
    assert state["fallback_used"] is False
    assert state["random_initialization_used"] is False


def test_zero_matched_vision_checkpoint_cannot_pass(tmp_path: Path, monkeypatch):
    checkpoint = tmp_path / "vision" / "checkpoint.pt"
    checkpoint.parent.mkdir()
    torch.save({"state_dict": {"unrelated.weight": torch.ones(2, 2)}}, checkpoint)
    monkeypatch.setitem(sys.modules, "open_clip", _manual_open_clip())

    wrapper = CLIPWrapper(prefer_pretrained=True, checkpoint_path=checkpoint, asset_mode="local_checkpoint")
    state = wrapper.readiness_state()

    assert state["matched_parameter_key_count"] == 0
    assert state["weights_loaded"] is False
    assert state["compatibility_failure_reason"] == "checkpoint_key_mismatch_zero_matched"


def test_partial_low_coverage_vision_checkpoint_fails(tmp_path: Path, monkeypatch):
    checkpoint = tmp_path / "vision" / "checkpoint.pt"
    checkpoint.parent.mkdir()
    torch.save({"state_dict": {"proj.bias": torch.ones(2)}}, checkpoint)
    monkeypatch.setitem(sys.modules, "open_clip", _manual_open_clip())

    wrapper = CLIPWrapper(prefer_pretrained=True, checkpoint_path=checkpoint, asset_mode="local_checkpoint")
    state = wrapper.readiness_state()

    assert 0.0 < state["matched_parameter_ratio"] < 0.99
    assert state["weights_loaded"] is False
    assert str(state["compatibility_failure_reason"]).startswith("checkpoint_key_mismatch_low_coverage")


def test_known_prefix_normalization_allows_valid_manual_checkpoint(tmp_path: Path, monkeypatch):
    checkpoint = tmp_path / "vision" / "checkpoint.pt"
    checkpoint.parent.mkdir()
    torch.save(
        {
            "model_name": "ViT-B-32",
            "state_dict": {
                "module.proj.weight": torch.ones(2, 2),
                "module.proj.bias": torch.ones(2),
            },
        },
        checkpoint,
    )
    monkeypatch.setitem(sys.modules, "open_clip", _manual_open_clip())

    wrapper = CLIPWrapper(prefer_pretrained=True, checkpoint_path=checkpoint, asset_mode="local_checkpoint")
    state = wrapper.readiness_state()

    assert state["weights_loaded"] is True
    assert state["checkpoint_compatibility_verified"] is True
    assert state["checkpoint_format"] == "manual_state_dict_validated"
    assert state["matched_parameter_ratio"] == 1.0
    assert state["checkpoint_model_name"] == "ViT-B-32"


def test_text_local_directory_runtime_with_mocked_transformers(tmp_path: Path, monkeypatch):
    text_dir = tmp_path / "text"
    text_dir.mkdir()
    _write_text_snapshot(text_dir, weight_name="pytorch_model.bin")
    _FakeTokenizer.last_kwargs = None
    fake_transformers = SimpleNamespace(AutoTokenizer=_FakeTokenizer, AutoModel=_FakeTextModel)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    wrapper = TextEncoderWrapper(
        prefer_transformers=True,
        checkpoint_path=text_dir,
        asset_mode="local_directory",
        allow_download=False,
        local_files_only=True,
    )
    state = wrapper.readiness_state()
    assert state["weights_loaded"] is True
    assert state["tokenizer_loaded"] is True
    assert state["tokenizer_use_fast"] is False
    assert state["tokenizer_backend_policy"] == "sentencepiece_slow"
    assert state["tokenizer_class"] == "_FakeTokenizer"
    assert state["fallback_used"] is False
    assert state["weights_source"] == "local_directory"
    assert _FakeTokenizer.last_kwargs["use_fast"] is False


def test_text_sentencepiece_available_success_readiness(tmp_path: Path, monkeypatch):
    text_dir = tmp_path / "text"
    text_dir.mkdir()
    _write_text_snapshot(text_dir, weight_name="pytorch_model.bin")
    _FakeTokenizer.last_kwargs = None
    monkeypatch.setattr("module.backbone.text._sentencepiece_available", lambda: True)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=_FakeTokenizer, AutoModel=_FakeTextModel))

    wrapper = TextEncoderWrapper(
        prefer_transformers=True,
        model_name="microsoft/deberta-v3-base",
        checkpoint_path=text_dir,
        asset_mode="local_directory",
        tokenizer_use_fast=False,
        tokenizer_backend_policy="sentencepiece_slow",
    )
    state = wrapper.readiness_state()

    assert _FakeTokenizer.last_kwargs["use_fast"] is False
    assert state["sentencepiece_required"] is True
    assert state["sentencepiece_available"] is True
    assert state["tokenizer_loaded"] is True
    assert state["weights_loaded"] is True
    assert state["fallback_used"] is False


def test_text_asset_verification_propagates_tokenizer_policy(tmp_path: Path, monkeypatch):
    text_dir = tmp_path / "text"
    text_dir.mkdir()
    _write_text_snapshot(text_dir, weight_name="pytorch_model.bin")
    _FakeTokenizer.last_kwargs = None
    monkeypatch.setattr("module.backbone.text._sentencepiece_available", lambda: True)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=_FakeTokenizer, AutoModel=_FakeTextModel))

    result = verify_pretrained_assets(_config(tmp_path, text_path=text_dir), strict=False)

    assert _FakeTokenizer.last_kwargs["use_fast"] is False
    assert result.text["runtime"]["tokenizer_use_fast"] is False
    assert result.text["runtime"]["tokenizer_backend_policy"] == "sentencepiece_slow"


def test_tokenizer_failure_cannot_be_masked_by_model_success(tmp_path: Path, monkeypatch):
    text_dir = tmp_path / "text"
    text_dir.mkdir()
    _write_text_snapshot(text_dir, weight_name="pytorch_model.bin")
    _CountingTextModel.called = False
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=_FailingTokenizer, AutoModel=_CountingTextModel))

    wrapper = TextEncoderWrapper(
        prefer_transformers=True,
        model_name="microsoft/deberta-v3-base",
        checkpoint_path=text_dir,
        asset_mode="local_directory",
    )
    state = wrapper.readiness_state()

    assert state["tokenizer_loaded"] is False
    assert state["weights_loaded"] is False
    assert state["fallback_used"] is True
    assert _CountingTextModel.called is False


def test_smoke_fallback_remains_available_but_strict_cannot_pass(tmp_path: Path):
    smoke = verify_pretrained_assets(_config(tmp_path), strict=False)
    strict = verify_pretrained_assets(_config(tmp_path), strict=True)
    assert smoke.passed is True
    assert smoke.warnings
    assert strict.passed is False


def test_assets_cli_help_exposes_nested_commands():
    result = subprocess.run(
        [sys.executable, "scripts/run.py", "assets", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    for word in ["inspect", "init-layout", "verify"]:
        assert word in result.stdout


def test_manifest_asset_provenance_does_not_infer_weights_loaded_from_manifest(tmp_path: Path):
    checkpoint = tmp_path / "vision" / "checkpoint.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    text_dir = tmp_path / "text"
    text_dir.mkdir()
    _write_text_snapshot(text_dir)
    cfg = _config(tmp_path, vision_path=checkpoint, text_path=text_dir)
    provenance = build_asset_provenance(
        cfg,
        runtime_state={
            "vision": {"weights_loaded": False, "fallback_used": True, "weights_source": None},
            "text": {
                "weights_loaded": False,
                "fallback_used": True,
                "weights_source": None,
                "tokenizer_use_fast": False,
                "tokenizer_backend_policy": "sentencepiece_slow",
                "tokenizer_class": "DebertaV2Tokenizer",
                "tokenizer_loaded": True,
                "sentencepiece_required": True,
                "sentencepiece_available": False,
            },
        },
    )
    manifest = build_run_manifest(
        suite_name=None,
        run_kind="ours_full",
        run_name="ours_full",
        dataset="harm_c",
        seed=42,
        config_path="configs/config.yaml",
        split_file=None,
        extra={"pretrained_asset_provenance": provenance},
    )
    assert manifest["pretrained_asset_provenance"]["vision"]["sha256"]
    assert manifest["pretrained_asset_provenance"]["vision"]["weights_loaded"] is False
    assert manifest["pretrained_asset_provenance"]["text"]["weights_loaded"] is False
    assert manifest["pretrained_asset_provenance"]["text"]["tokenizer_use_fast"] is False
    assert manifest["pretrained_asset_provenance"]["text"]["tokenizer_backend_policy"] == "sentencepiece_slow"
    assert manifest["pretrained_asset_provenance"]["text"]["tokenizer_loaded"] is True
    assert manifest["pretrained_asset_provenance"]["text"]["sentencepiece_required"] is True
    assert manifest["pretrained_asset_provenance"]["text"]["sentencepiece_available"] is False


def _config(tmp_path: Path, *, vision_path: Path | None = None, text_path: Path | None = None) -> dict:
    return {
        "runtime": {"device": "cpu"},
        "backbone": {
            "clip": {
                "prefer_pretrained": True,
                "model_name": "ViT-B-32",
                "asset_mode": "local_checkpoint",
                "checkpoint_path": str(vision_path or tmp_path / "missing_vision.pt"),
                "cache_dir": str(tmp_path / "vision_cache"),
                "local_files_only": True,
                "allow_download": False,
            },
            "text": {
                "prefer_transformers": True,
                "model_name": "microsoft/deberta-v3-base",
                "asset_mode": "local_directory",
                "checkpoint_path": str(text_path or tmp_path / "missing_text"),
                "cache_dir": str(tmp_path / "text_cache"),
                "tokenizer_use_fast": False,
                "tokenizer_backend_policy": "sentencepiece_slow",
                "local_files_only": True,
                "allow_download": False,
            },
        },
    }


def _write_text_snapshot(path: Path, weight_name: str = "model.safetensors") -> None:
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "special_tokens_map.json").write_text("{}", encoding="utf-8")
    (path / weight_name).write_bytes(b"weights")


def _fake_clip_tuple(calls: dict, pretrained):
    calls["pretrained"] = pretrained
    return _FakeClipModel(), None, lambda image: torch.zeros(3, 4, 4)


class _FakeClipModel(torch.nn.Module):
    def encode_image(self, tensor):
        return torch.ones(tensor.size(0), 4)


class _ParamClipModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2)

    def encode_image(self, tensor):
        return torch.ones(tensor.size(0), 2)


def _manual_open_clip():
    def create_model_and_transforms(model_name, pretrained=None, cache_dir=None):
        if pretrained is not None:
            raise RuntimeError("factory local path unsupported in test")
        return _ParamClipModel(), None, lambda image: torch.zeros(3, 4, 4)

    return SimpleNamespace(create_model_and_transforms=create_model_and_transforms)


class _FakeTokenizer:
    last_kwargs = None

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        assert kwargs.get("local_files_only") is True
        cls.last_kwargs = dict(kwargs)
        return cls()

    def __call__(self, text, **kwargs):
        return _Encoded({"input_ids": torch.tensor([[1, 2]])})

    def convert_ids_to_tokens(self, ids):
        return [str(item) for item in ids]


class _Encoded(dict):
    def to(self, device):
        return self


class _FakeTextModel(torch.nn.Module):
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        assert kwargs.get("local_files_only") is True
        return cls()

    def forward(self, **kwargs):
        return SimpleNamespace(last_hidden_state=torch.ones(1, 2, 4))


class _FailingTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise RuntimeError("tokenizer load failed")


class _CountingTextModel(_FakeTextModel):
    called = False

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.called = True
        return cls()
