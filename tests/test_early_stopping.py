from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from dataset import MemeDataset
from experiments.early_stopping import EarlyStopping, save_checkpoint, structured_validation_score
from experiments.train import BaselineRunConfig, run_baseline_experiment


def test_early_stopping_stops_after_patience():
    stopper = EarlyStopping(patience=2, min_delta=0.01, mode="max")
    assert stopper.step(0.5, epoch=1)["is_best"]
    assert not stopper.step(0.505, epoch=2)["stopped_early"]
    assert stopper.step(0.506, epoch=3)["stopped_early"]
    assert stopper.best_epoch == 1


def test_best_checkpoint_payload_is_saved(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = save_checkpoint(
        tmp_path / "best_model.pt",
        epoch=3,
        model=model,
        optimizer=optimizer,
        best_metric=0.8,
        config={"name": "toy"},
        seed=42,
        dataset="harm_c",
        model_name="toy_model",
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 3
    assert checkpoint["best_metric"] == 0.8
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint


def test_empty_validation_split_does_not_crash(tmp_path):
    source = _write_tiny_dataset(tmp_path)
    config_path = _write_config(tmp_path, source)
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"train": ["s0", "s1"], "valid": [], "test": ["s2", "s3"]}), encoding="utf-8")
    result_root = tmp_path / "result"
    metrics = run_baseline_experiment(
        BaselineRunConfig(
            model_name="text_only_encoder",
            dataset_name="harm_c",
            seed=42,
            config_path=str(config_path),
            split_file=str(split_path),
            output_root=str(result_root),
            epochs=2,
            batch_size=2,
            patience=1,
            min_delta=0.1,
            device="cpu",
        )
    )
    assert "macro_f1" in metrics
    log_path = result_root / "predictions" / "harm_c" / "text_only_encoder" / "42" / "training_log.json"
    assert log_path.exists()
    log = json.loads(log_path.read_text(encoding="utf-8"))
    assert len(log) == 2
    assert log[-1]["stopped_early"] is False


def test_training_clis_accept_early_stopping_flags():
    baseline = subprocess.run(
        [
            sys.executable,
            "scripts/run.py",
            "baseline",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--min-delta" in baseline.stdout
    assert "--early-stop-metric" in baseline.stdout
    assert "--disable-tqdm" in baseline.stdout

    ours = subprocess.run(
        [
            sys.executable,
            "scripts/run.py",
            "train",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--patience" in ours.stdout
    assert "--early-stop-metric" in ours.stdout
    assert "--disable-tqdm" in ours.stdout
    assert "--print-components" in ours.stdout


def test_structured_validation_score_skips_missing_components():
    score = structured_validation_score(
        {
            "macro_f1": 0.4,
            "harmfulness_macro_f1": 0.6,
            "target_granularity_macro_f1": 0.8,
            "intent_primary_macro_f1": None,
        }
    )
    assert score == 0.7


def _write_config(tmp_path: Path, source: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
paths:
  dataset_root: {source}
  annotation_root: {tmp_path / 'annotation'}
model:
  hidden_dim: 256
backbones:
  clip:
    prefer_pretrained: false
  text:
    prefer_transformers: false
runtime:
  device: cpu
""",
        encoding="utf-8",
    )
    return path


def _write_tiny_dataset(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    root = source / "covid_img+text"
    (root / "txt").mkdir(parents=True)
    (root / "img").mkdir(parents=True)
    records = []
    for idx in range(4):
        records.append({"id": f"s{idx}", "image": f"s{idx}.png", "labels": idx % 2, "text": f"sample {idx}"})
        (root / "img" / f"s{idx}.png").write_bytes(b"not-real")
    (root / "txt" / "all.jsonl").write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    ann = tmp_path / "annotation" / "harm_c"
    ann.mkdir(parents=True)
    (ann / "harmc_annotations.jsonl").write_text(
        "\n".join(json.dumps({"sample_id": row["id"], "annotation": {}}) for row in records) + "\n",
        encoding="utf-8",
    )
    _ = MemeDataset(dataset_root=source, annotation_root=tmp_path / "annotation", dataset_names=["harm_c"], keep_missing_images=True)
    return source
