from __future__ import annotations

import json
from pathlib import Path
import torch
from dataset import MemeDataset
from experiments.reporting import write_aggregate_tables
from experiments.data_preparation import compute_dataset_statistics
from experiments.evaluation import compute_harmfulness_metrics
from experiments.splits import build_splits_for_dataset, label_to_int
from experiments.train import BaselineRunConfig, run_baseline_experiment
from module.baseline import CLIPTextConcatClassifier, ImageOnlyCLIPClassifier, TextOnlyEncoderClassifier
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.ablation_configs import ABLATION_MODES, FUSION_MODES, KNOWLEDGE_MODES, get_ablation_config
from experiments.ablation_runner import execute_variant_pipeline
from experiments.knowledge_comparison import run_knowledge_comparison
from experiments.evaluation import evaluate_structured_predictions, write_structured_aggregate_tables
from experiments.train import OursRunConfig, run_ours_experiment
from module.runner import HarmfulMemePipeline
from module.losses import (
    StructuredMemeLoss,
    classification_loss_from_logits,
    extract_supervision_from_annotation,
    multilabel_loss_from_logits,
)
from utils.eval_utils import binary_classification_metrics, evidence_precision_recall_at_k, multiclass_metrics


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


def test_structured_evaluator_on_fake_predictions():
    records = [
        {
            "gold_label": 1,
            "pred_label": 1,
            "prob_harmful": 0.9,
            "gold_target": {"target_presence": "explicit", "target_granularity": "community", "protected_attribute": ["nationality"]},
            "target": {"presence": "explicit", "granularity": "community", "attributes": ["nationality"]},
            "gold_intent": {"intent_primary": "ridicule_mockery", "stance": "hostile", "background_knowledge_needed": True},
            "intent": {"primary": "ridicule_mockery", "stance": "hostile", "background_knowledge_needed": True},
            "gold_tactic": {"tactic_rhetorical": ["sarcasm_irony"], "tactic_multimodal_relation": "incongruent"},
            "tactic": {"rhetorical": ["sarcasm_irony"], "multimodal_relation": "incongruent"},
            "gold_evidence_text": ["mocking caption"],
            "supporting_evidence": {"internal": [{"text": "mocking caption", "score": 0.8}], "external": []},
        }
    ]
    metrics = evaluate_structured_predictions(records)
    assert metrics["harmfulness_accuracy"] == 1.0
    assert metrics["target_presence_macro_f1"] == 1.0
    assert metrics["evidence_hit_at_k"] == 1.0


def test_ablation_knowledge_and_fusion_modes_do_not_crash():
    pipeline = HarmfulMemePipeline().eval()
    sample = _sample()
    for name in ABLATION_MODES:
        outputs = execute_variant_pipeline(pipeline, sample, ablation=get_ablation_config(name))
        assert outputs["stage_e"].structured_prediction["sample_id"] == "s1"
    assert get_ablation_config("w_o_verifier").name == "w_o_support_verifier"
    for mode in KNOWLEDGE_MODES:
        outputs = execute_variant_pipeline(pipeline, sample, knowledge_mode=mode)
        assert "stage_c" in outputs
    for mode in FUSION_MODES:
        outputs = execute_variant_pipeline(pipeline, sample, fusion_mode=mode)
        assert outputs["stage_d"].metadata.gate_mode == mode


def test_ours_and_knowledge_runners_on_tiny_dataset(tmp_path):
    source = _write_tiny_dataset(tmp_path)
    config_path = _write_config(tmp_path, source)
    result_root = tmp_path / "result"

    metrics = run_ours_experiment(
        OursRunConfig(
            dataset_name="harm_c",
            seed=42,
            config_path=str(config_path),
            output_root=str(result_root),
            epochs=0,
            device="cpu",
        )
    )
    assert "macro_f1" in metrics
    assert (result_root / "predictions" / "harm_c" / "ours_full" / "42" / "final_predictions.jsonl").exists()

    knowledge_metrics = run_knowledge_comparison(
        "harm_c",
        "no_knowledge",
        seed=42,
        config_path=str(config_path),
        output_root=str(result_root),
    )
    assert "accepted_knowledge_count" in knowledge_metrics

    per_run, mean_std = write_structured_aggregate_tables(result_root / "predictions", result_root / "metrics")
    assert per_run.exists()
    assert mean_std.exists()


def _sample() -> dict:
    return {
        "sample_id": "s1",
        "dataset_name": "harm_c",
        "image_path": None,
        "ocr_text_full": "THIS MEME MOCKS AMERICA WITH A SARCASTIC CAPTION",
        "raw_label": 1,
        "annotation": {
            "target": {"target_presence": "explicit", "target_granularity": "community", "protected_attribute": ["nationality"]},
            "intent": {"intent_primary": "ridicule_mockery", "stance": "hostile", "background_knowledge_needed": True},
            "tactic": {"tactic_rhetorical": ["sarcasm_irony"], "tactic_multimodal_relation": "incongruent"},
            "evidence": {"key_text_evidence": "sarcastic caption"},
        },
    }


def _write_config(tmp_path: Path, source: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
paths:
  dataset_root: {source}
  annotation_root: {tmp_path / 'annotation'}
  retrieval_corpus_paths: []
model:
  hidden_dim: 256
  knowledge_top_k: 4
backbones:
  clip:
    prefer_pretrained: false
    model_name: ViT-B-32
  text:
    prefer_transformers: false
    model_name: microsoft/deberta-v3-base
  retriever:
    fallback_candidates: true
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
    for idx in range(6):
        label = idx % 2
        records.append({"id": f"s{idx}", "image": f"s{idx}.png", "labels": label, "text": f"sample {idx} sarcastic label {label}"})
        (root / "img" / f"s{idx}.png").write_bytes(b"not-real")
    for name, subset in {"all": records, "train": records[:4], "val": records[4:5], "test": records[5:]}.items():
        (root / "txt" / f"{name}.jsonl").write_text("\n".join(json.dumps(row) for row in subset) + "\n", encoding="utf-8")
    annotation_root = tmp_path / "annotation" / "harm_c"
    annotation_root.mkdir(parents=True)
    (annotation_root / "harmc_annotations.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "sample_id": row["id"],
                    "dataset_name": "harm_c",
                    "annotation": {
                        "target": {"target_presence": "explicit", "target_granularity": "community"},
                        "intent": {"intent_primary": "ridicule_mockery", "stance": "hostile", "background_knowledge_needed": False},
                        "tactic": {"tactic_rhetorical": ["sarcasm_irony"], "tactic_multimodal_relation": "text_visual_overlap"},
                    },
                }
            )
            for row in records
        )
        + "\n",
        encoding="utf-8",
    )
    return source


def test_phase2_losses_and_metrics_smoke():
    pipeline = HarmfulMemePipeline().eval()
    outputs = pipeline(
        {
            "sample_id": "s1",
            "dataset_name": "harm_c",
            "image_path": None,
            "ocr_text_full": "THIS IS A SHIT SHOW ABOUT AMERICA",
        }
    )
    losses = StructuredMemeLoss()(outputs["stage_e"], {"harmfulness": "harmful"})
    assert "total" in losses
    assert losses["total"].item() >= 0
    assert losses["harmfulness"].requires_grad

    binary = binary_classification_metrics([1, 0, 1], [1, 0, 0])
    assert binary["accuracy"] >= 0
    multi = multiclass_metrics(["a", "b"], ["a", "a"])
    assert "confusion" in multi
    evidence = evidence_precision_recall_at_k(["e1", "e2"], {"e2"}, k=2)
    assert evidence["recall_at_k"] == 1.0


def test_nested_annotation_supervision_and_logits_losses():
    annotation = {
        "raw_label": 1,
        "annotation": {
            "target": {"target_presence": "explicit", "target_granularity": "community", "protected_attribute": ["nationality"]},
            "intent": {"intent_primary": "ridicule_mockery", "stance": "hostile", "secondary_intent": "self_expression", "background_knowledge_needed": True},
            "tactic": {"tactic_rhetorical": ["sarcasm_irony"], "tactic_multimodal_relation": "incongruent"},
            "evidence": {"key_text_evidence": "mocking text", "key_visual_evidence": "visual cue"},
        },
    }
    supervision = extract_supervision_from_annotation(annotation)
    assert supervision["harmfulness"] == "harmful"
    assert supervision["target_granularity"] == "community"
    assert supervision["tactic_rhetorical"] == ["sarcasm_irony"]
    assert supervision["evidence_text"] == ["mocking text", "visual cue"]

    logits = torch.randn(3, requires_grad=True)
    loss = classification_loss_from_logits(logits, 1)
    assert loss.requires_grad
    multi = multilabel_loss_from_logits(logits, [0, 2])
    assert multi.requires_grad
