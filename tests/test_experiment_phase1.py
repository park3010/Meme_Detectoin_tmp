from __future__ import annotations

import json
from pathlib import Path

import torch

from dataset import MemeDataset
from experiments.aggregate_results import write_aggregate_tables
from experiments.dataset_stats import compute_dataset_statistics
from experiments.evaluation import compute_harmfulness_metrics
from experiments.splits import build_splits_for_dataset, label_to_int
from experiments.train import BaselineRunConfig, run_baseline_experiment
from module.baseline import CLIPTextConcatClassifier, ImageOnlyCLIPClassifier, TextOnlyEncoderClassifier


def test_metrics_handle_auc_edge_cases():
    metrics = compute_harmfulness_metrics([1, 1], [1, 0], [0.8, 0.3])
    assert metrics["roc_auc"] is None
    assert metrics["tp"] == 1

    metrics = compute_harmfulness_metrics([0, 1], [0, 1], [0.1, 0.9])
    assert metrics["roc_auc"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_baseline_forward_shapes():
    texts = ["hello meme", "harmful text"]
    images = [None, None]
    for model in [ImageOnlyCLIPClassifier(), TextOnlyEncoderClassifier(), CLIPTextConcatClassifier()]:
        output = model(image_paths=images, ocr_texts=texts)
        assert output["logits"].shape == (2, 2)
        assert output["prob_harmful"].shape == (2,)


def test_split_generation_uses_official_files(tmp_path):
    source = _write_tiny_dataset(tmp_path)
    dataset = MemeDataset(dataset_root=source, annotation_root=tmp_path / "annotation", dataset_names=["harm_c"], keep_missing_images=True)
    splits = build_splits_for_dataset("harm_c", dataset, seed=42, dataset_root=source)
    assert splits["train"] == ["s0", "s1", "s2", "s3"]
    assert splits["valid"] == ["s4"]
    assert splits["test"] == ["s5"]
    assert label_to_int("harmful") == 1


def test_dataset_stats_on_tiny_dataset(tmp_path):
    source = _write_tiny_dataset(tmp_path)
    rows = compute_dataset_statistics(source, tmp_path / "annotation", ["harm_c"])
    assert rows[0]["dataset_name"] == "harm_c"
    assert rows[0]["total_samples"] == 6
    assert rows[0]["harmful_count"] == 3


def test_tiny_text_baseline_training_and_aggregation(tmp_path):
    source = _write_tiny_dataset(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
paths:
  dataset_root: {source}
  annotation_root: {tmp_path / 'annotation'}
  result_root: {tmp_path / 'result'}
model:
  hidden_dim: 256
backbones:
  clip:
    prefer_pretrained: false
    model_name: ViT-B-32
  text:
    prefer_transformers: false
    model_name: microsoft/deberta-v3-base
""",
        encoding="utf-8",
    )
    metrics = run_baseline_experiment(
        BaselineRunConfig(
            model_name="text_only_encoder",
            dataset_name="harm_c",
            seed=42,
            config_path=str(config_path),
            output_root=str(tmp_path / "result"),
            epochs=1,
            batch_size=2,
            lr=1e-3,
            patience=1,
            device="cpu",
        )
    )
    assert "macro_f1" in metrics
    pred_path = tmp_path / "result" / "predictions" / "harm_c" / "text_only_encoder" / "42" / "final_predictions.jsonl"
    assert pred_path.exists()

    main_path, mean_std_path = write_aggregate_tables(tmp_path / "result" / "predictions", tmp_path / "result" / "metrics")
    assert main_path.exists()
    assert mean_std_path.exists()


def _write_tiny_dataset(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    root = source / "covid_img+text"
    (root / "txt").mkdir(parents=True)
    (root / "img").mkdir(parents=True)
    records = []
    for idx in range(6):
        label = idx % 2
        records.append({"id": f"s{idx}", "image": f"s{idx}.png", "labels": label, "text": f"sample {idx} label {label}"})
        (root / "img" / f"s{idx}.png").write_bytes(b"not-real")
    for name, subset in {
        "all": records,
        "train": records[:4],
        "val": records[4:5],
        "test": records[5:],
    }.items():
        (root / "txt" / f"{name}.jsonl").write_text("\n".join(json.dumps(row) for row in subset) + "\n", encoding="utf-8")
    annotation_root = tmp_path / "annotation" / "harm_c"
    annotation_root.mkdir(parents=True)
    (annotation_root / "harmc_annotations.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "sample_id": row["id"],
                    "dataset_name": "harmc",
                    "annotation": {
                        "target": {"target_granularity": "community"},
                        "intent": {"intent_primary": "entertainment", "background_knowledge_needed": False},
                        "tactic": {"tactic_rhetorical": ["other"]},
                    },
                }
            )
            for row in records
        )
        + "\n",
        encoding="utf-8",
    )
    return source
