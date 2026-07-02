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
    assert tactic["prediction_source"] == "logits_only"
    assert tactic["rendered_label_field_forbidden_for_metric"] is True
    assert tactic["implementation_status"] == "blocked"
    json.dumps(contract)


def test_preflight_cli_help_exposes_required_options():
    result = subprocess.run([sys.executable, "scripts/run.py", "preflight", "--help"], cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)

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
