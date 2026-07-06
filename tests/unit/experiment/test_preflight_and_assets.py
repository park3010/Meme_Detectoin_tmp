from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
from experiments.metric_contract import resolve_metric_contract
from experiments.preflight import (
    inspect_backbone_readiness,
    inspect_dataset_metric_eligibility,
    inspect_retrieval_corpus_readiness,
    inspect_split_integrity,
    run_preflight,
)
from module.backbone.text import TextEncoderWrapper
from module.backbone.vision import CLIPWrapper
from utils.io import load_yaml
from experiments.pretrained_assets import (
    build_asset_provenance,
    inspect_text_asset,
    inspect_vision_asset,
    verify_pretrained_assets,
    write_asset_manifest,
)
from experiments.run_manifest import build_run_manifest


def test_backbone_state_honesty_and_profile_blocking(tmp_path: Path, monkeypatch):
    cfg = _tiny_config(tmp_path)
    cfg["backbone"]["clip"]["prefer_pretrained"] = True
    cfg["backbone"]["text"]["prefer_transformers"] = True

    smoke = run_preflight(
        profile="smoke",
        config_path=_write_config(tmp_path, cfg),
        datasets=["harm_c"],
        seeds=[42],
        output_root=tmp_path / "result",
        normalized_root=tmp_path / "normalized",
        vocab_path="configs/label_vocab.yaml",
        write_report=True,
    )
    assert smoke.passed is True
    assert any(item["code"] in {"vision_fallback_backbone", "text_fallback_backbone"} for item in smoke.warnings)

    main = run_preflight(
        profile="main_experiment",
        config_path=_write_config(tmp_path, cfg),
        datasets=["harm_c"],
        seeds=[42],
        output_root=tmp_path / "result_main",
        normalized_root=tmp_path / "normalized",
        vocab_path="configs/label_vocab.yaml",
        write_report=True,
        strict=True,
    )
    assert main.passed is False
    assert any(item["code"].endswith("pretrained_missing") or item["code"].endswith("fallback_backbone") for item in main.errors)

    fake_open_clip = SimpleNamespace(create_model_and_transforms=lambda *args, **kwargs: (_FakeVisionModel(), None, lambda image: torch.zeros(3, 4, 4)))
    monkeypatch.setitem(sys.modules, "open_clip", fake_open_clip)
    random_clip = CLIPWrapper(prefer_pretrained=True, local_files_only=False, allow_download=True, pretrained_tag=None)
    state = random_clip.readiness_state()
    assert state["resolved_backend"] == "open_clip"
    assert state["weights_loaded"] is False
    assert state["random_initialization_used"] is True

    checkpoint = tmp_path / "vision.pt"
    torch.save({}, checkpoint)
    loaded_clip = CLIPWrapper(prefer_pretrained=True, checkpoint_path=checkpoint)
    assert loaded_clip.readiness_state()["weights_loaded"] is True


def test_text_fallback_and_mocked_transformer_readiness(tmp_path: Path, monkeypatch):
    fallback = TextEncoderWrapper(prefer_transformers=True, model_name="missing-local-model")
    assert fallback.readiness_state()["weights_loaded"] is False
    assert fallback.readiness_state()["fallback_used"] is True

    fake_transformers = SimpleNamespace(AutoTokenizer=_FakeTokenizer, AutoModel=_FakeTextModel)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    checkpoint_dir = tmp_path / "hf"
    checkpoint_dir.mkdir()
    loaded = TextEncoderWrapper(prefer_transformers=True, checkpoint_path=checkpoint_dir)
    state = loaded.readiness_state()
    assert state["weights_loaded"] is True
    assert state["fallback_used"] is False


def test_strict_main_preflight_reports_missing_sentencepiece(tmp_path: Path, monkeypatch):
    cfg = _tiny_config(tmp_path)
    text_dir = tmp_path / "text_snapshot"
    text_dir.mkdir()
    _write_text_snapshot(text_dir)
    cfg["backbone"]["text"].update(
        {
            "model_name": "microsoft/deberta-v3-base",
            "asset_mode": "local_directory",
            "checkpoint_path": str(text_dir),
            "tokenizer_use_fast": False,
            "tokenizer_backend_policy": "sentencepiece_slow",
        }
    )
    monkeypatch.setattr("module.backbone.text._sentencepiece_available", lambda: False)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=_FakeTokenizer, AutoModel=_FakeTextModel))

    result = run_preflight(
        profile="main_experiment",
        config_path=_write_config(tmp_path, cfg),
        datasets=["harm_c"],
        seeds=[42],
        output_root=tmp_path / "result_sentencepiece",
        normalized_root=tmp_path / "normalized",
        vocab_path="configs/label_vocab.yaml",
        write_report=True,
        strict=True,
    )

    codes = {item["code"] for item in result.errors}
    assert "text_sentencepiece_dependency_missing" in codes
    report = Path(result.artifacts["preflight_report"]).read_text(encoding="utf-8")
    assert "python -m pip install sentencepiece" in report


def test_dataset_eligibility_ignores_ambiguous_unknown_and_detects_ineligible(tmp_path: Path):
    split = tmp_path / "result" / "splits" / "harm_c" / "seed_42.json"
    split.parent.mkdir(parents=True)
    split.write_text(json.dumps({"train": ["s1"], "valid": ["s2"], "test": ["s3", "s4"]}), encoding="utf-8")
    _write_normalized(
        tmp_path / "normalized" / "harm_c" / "normalized_clean.jsonl",
        [
            ("s1", {"harmfulness": "harmful", "target_presence": "explicit", "target_granularity": "community", "intent_primary": "ridicule_mockery", "tactic_multimodal_relation": "incongruent", "tactic_rhetorical": ["sarcasm_irony", "unknown"]}),
            ("s2", {"harmfulness": "non_harmful", "target_presence": "ambiguous", "target_granularity": "unknown", "intent_primary": "neutral", "tactic_multimodal_relation": "unknown", "tactic_rhetorical": ["unknown"]}),
            ("s3", {"harmfulness": "harmful", "target_presence": "explicit", "target_granularity": "community", "intent_primary": "ridicule_mockery", "tactic_multimodal_relation": "incongruent", "tactic_rhetorical": ["sarcasm_irony"]}),
            ("s4", {"harmfulness": "harmful", "target_presence": "unknown", "target_granularity": "ambiguous", "intent_primary": "ridicule_mockery", "tactic_multimodal_relation": "ambiguous", "tactic_rhetorical": ["unknown"]}),
        ],
    )
    issues = []
    cfg = load_yaml("configs/config.yaml")
    report, rows = inspect_dataset_metric_eligibility(
        cfg,
        ["harm_c"],
        [42],
        split_report={"harm_c": {"42": {"path": str(split)}}},
        normalized_root=tmp_path / "normalized",
        label_set="clean",
        vocab_path="configs/label_vocab.yaml",
        issues=issues,
        require_metric_eligibility=True,
    )

    target_valid = report["harm_c"]["42"]["target_presence"]["splits"]["valid"]
    assert target_valid["ignored_ambiguous_count"] == 1
    target_test = report["harm_c"]["42"]["target_presence"]["splits"]["test"]
    assert target_test["ignored_unknown_count"] == 1
    assert report["harm_c"]["42"]["target_presence"]["eligible_hard"] is False
    rhetorical_test = report["harm_c"]["42"]["tactic_rhetorical"]["splits"]["test"]
    assert rhetorical_test["number_of_observed_non_ignored_classes"] == 1
    assert rows
    json.dumps(report)


def test_split_integrity_detects_overlap_duplicates_and_sha(tmp_path: Path):
    source = _write_source_dataset(tmp_path)
    split = tmp_path / "result" / "splits" / "harm_c" / "seed_42.json"
    split.parent.mkdir(parents=True)
    split.write_text(json.dumps({"train": ["s1", "s1"], "valid": ["s1", "s2"], "test": ["s3"]}), encoding="utf-8")
    issues = []
    report = inspect_split_integrity(
        {"paths": {"dataset_root": str(source), "annotation_root": str(tmp_path / "annotation")}},
        ["harm_c"],
        [42],
        output_root=tmp_path / "result",
        create_missing_splits=False,
        overwrite_splits=False,
        issues=issues,
    )

    assert report["harm_c"]["42"]["sha256"]
    assert any(issue.code == "split_duplicate_ids" for issue in issues)
    assert any(issue.code == "split_overlap" for issue in issues)


def test_retrieval_corpus_audit_counts_missing_empty_and_valid(tmp_path: Path):
    missing = tmp_path / "missing.jsonl"
    empty = tmp_path / "empty.jsonl"
    valid = tmp_path / "valid.jsonl"
    empty.write_text("", encoding="utf-8")
    valid.write_text(json.dumps({"id": "d1", "text": "usable text"}) + "\n" + json.dumps({"id": "d2", "text": ""}) + "\n", encoding="utf-8")
    issues = []
    report = inspect_retrieval_corpus_readiness(
        {
            "paths": {"retrieval_corpus_paths": [str(missing), str(empty), str(valid)]},
            "backbone": {"retriever": {"fallback_candidates": True}},
            "stages": {"stage_b": {"top_k": 8}, "stage_c": {"min_relevance": 0.05, "allow_low_relevance_fallback": True}},
        },
        {"require_retrieval_corpus": True},
        issues,
    )
    assert report["usable_corpus_count"] == 1
    assert report["paths"][0]["exists"] is False
    assert report["paths"][2]["parseable_record_count"] == 2
    assert "fallback candidate != retrieved external knowledge" in report["policy"]["semantic_note"]


def test_metric_contract_forbids_rendered_rhetorical_labels():
    contract = resolve_metric_contract(load_yaml("configs/config.yaml"), vocab_path="configs/label_vocab.yaml")

    tactic = contract["fields"]["tactic_rhetorical"]
    assert tactic["prediction_source"] == "tactic_logits_sigmoid"
    assert tactic["rendered_label_field_forbidden_for_metric"] is True
    assert tactic["none_label_policy"] == "fallback_when_no_non_none_label_selected"
    assert tactic["implementation_status"] == "ready"
    assert contract["implementation_status"] == "ready"
    json.dumps(contract)


def test_preflight_cli_help_exposes_required_options():
    result = subprocess.run([sys.executable, "scripts/run.py", "preflight", "--help"], cwd=Path(__file__).resolve().parents[3], text=True, capture_output=True)

    assert result.returncode == 0
    for option in ["--profile", "--create-missing-splits", "--no-create-missing-splits", "--probe-pipeline", "--allow-fallback", "--allow-download"]:
        assert option in result.stdout


class _FakeVisionModel(torch.nn.Module):
    def encode_image(self, tensor):
        return torch.ones(tensor.size(0), 4)


class _FakeTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
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
        return cls()

    def forward(self, **kwargs):
        return SimpleNamespace(last_hidden_state=torch.ones(1, 2, 4))


def _tiny_config(tmp_path: Path) -> dict:
    source = _write_source_dataset(tmp_path)
    return {
        "paths": {
            "dataset_root": str(source),
            "annotation_root": str(tmp_path / "annotation"),
            "retrieval_corpus_paths": [],
        },
        "model": {"hidden_dim": 256, "knowledge_top_k": 4},
        "backbone": {
            "clip": {"prefer_pretrained": True, "model_name": "ViT-B-32", "local_files_only": True, "allow_download": False},
            "text": {"prefer_transformers": True, "model_name": "missing-local-model", "local_files_only": True, "allow_download": False},
            "retriever": {"fallback_candidates": True},
        },
        "stages": {"stage_b": {"top_k": 4}, "stage_c": {"min_relevance": 0.05, "allow_low_relevance_fallback": True}},
        "preflight": load_yaml("configs/config.yaml")["preflight"],
    }


def _write_config(tmp_path: Path, cfg: dict) -> Path:
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _write_source_dataset(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    txt = source / "covid_img+text" / "txt"
    txt.mkdir(parents=True, exist_ok=True)
    rows = [
        {"id": "s1", "text": "one", "label": 1},
        {"id": "s2", "text": "two", "label": 0},
        {"id": "s3", "text": "three", "label": 1},
        {"id": "s4", "text": "four", "label": 1},
    ]
    txt.joinpath("all.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return source


def _write_normalized(path: Path, rows: list[tuple[str, dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for sample_id, labels in rows:
        records.append(
            {
                "sample_id": sample_id,
                "dataset_name": "harm_c",
                "image_path": None,
                "ocr_text_full": "",
                "raw_label": labels.get("harmfulness"),
                "labels": labels,
            }
        )
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def _write_text_snapshot(path: Path) -> None:
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "spm.model").write_bytes(b"spm")
    (path / "pytorch_model.bin").write_bytes(b"weights")


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
        cwd=Path(__file__).resolve().parents[3],
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
