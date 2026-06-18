from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ablation_configs import ABLATION_MODES, FUSION_MODES, KNOWLEDGE_MODES, get_ablation_config
from experiments.ablation_runner import execute_variant_pipeline
from experiments.knowledge_comparison import run_knowledge_comparison
from experiments.structured_eval import evaluate_structured_predictions, write_structured_aggregate_tables
from experiments.train_ours import OursRunConfig, run_ours_experiment
from module.pipeline.model import HarmfulMemePipeline


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
