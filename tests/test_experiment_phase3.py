from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cross_domain import CrossDomainConfig, run_cross_domain
from experiments.error_case_analysis import select_error_cases
from experiments.paper_tables import export_paper_tables
from experiments.rationale_eval import rationale_metrics, run_rationale_evaluation
from experiments.runtime_cost import run_runtime_cost_analysis
from experiments.significance import paired_significance_row, run_significance_tests
from experiments.subset_analysis import assign_subsets, run_subset_analysis
from experiments.verifier_eval import verifier_metrics
from utils.io import write_json, write_jsonl


def test_verifier_weak_label_metrics():
    examples = [
        {"weak_relevance_label": 1, "final_decision": "accepted", "relevance_score": 0.9, "weak_support_label": "support", "support_label": "support", "validity_score": 0.8},
        {"weak_relevance_label": 0, "final_decision": "rejected", "relevance_score": 0.1, "weak_support_label": "insufficient", "support_label": "insufficient", "validity_score": 0.1},
    ]
    metrics = verifier_metrics(examples)
    assert metrics["relevance_precision"] == 1.0
    assert metrics["support_macro_f1"] >= 0.0


def test_subset_error_rationale_significance_and_tables(tmp_path):
    result = tmp_path / "result"
    records = _fake_predictions()
    pred_path = result / "predictions" / "harm_c" / "ours_full" / "42" / "final_predictions.jsonl"
    write_jsonl(pred_path, records)
    write_json(result / "predictions" / "harm_c" / "ours_full" / "52" / "metrics.json", {"macro_f1": 0.7})
    write_json(result / "predictions" / "harm_c" / "ours_full" / "42" / "metrics.json", {"macro_f1": 0.8})
    write_json(result / "predictions" / "harm_c" / "text_only_encoder" / "42" / "metrics.json", {"macro_f1": 0.6})
    write_json(result / "predictions" / "harm_c" / "text_only_encoder" / "52" / "metrics.json", {"macro_f1": 0.65})
    _write_csv(result / "dataset_stats" / "dataset_statistics.csv", "dataset_name,total_samples\nharm_c,2\n")
    _write_csv(result / "metrics" / "main_performance.csv", "dataset,model,seed,macro_f1\nharm_c,ours_full,42,0.8\n")

    subsets = assign_subsets(records[0])
    assert "image_text_incongruity" in subsets
    assert run_subset_analysis("harm_c", "ours_full", 42, str(result))

    cases = select_error_cases("harm_c", "ours_full", 42, str(result))
    assert cases

    rationale = rationale_metrics(records[0])
    assert rationale["evidence_mention_rate"] is not None
    assert run_rationale_evaluation("harm_c", "ours_full", 42, str(result))

    sig = paired_significance_row(
        "harm_c",
        "macro_f1",
        "ours_full",
        {"42": {"macro_f1": 0.8}, "52": {"macro_f1": 0.7}},
        "text_only_encoder",
        {"42": {"macro_f1": 0.6}, "52": {"macro_f1": 0.65}},
    )
    assert sig["delta"] > 0
    assert run_significance_tests(str(result))

    tables = export_paper_tables(str(result))
    assert (result / "paper_tables" / "table1_dataset_statistics.csv").exists()
    assert len(tables) == 8


def test_cross_domain_and_runtime_on_tiny_dataset(tmp_path):
    source = _write_tiny_dataset(tmp_path)
    config = _write_config(tmp_path, source)
    result = tmp_path / "result"
    rows = run_cross_domain(
        CrossDomainConfig(
            setting="in_domain",
            train_dataset="harm_c",
            seed=42,
            config_path=str(config),
            output_root=str(result),
            limit=4,
            epochs=0,
        )
    )
    assert rows and rows[0]["test_dataset"] == "harm_c"
    summary = run_runtime_cost_analysis("harm_c", limit=1, warmup=0, config_path=str(config), output_root=str(result))
    assert summary["sample_count"] == 1
    assert "total_latency_ms" in summary


def _fake_predictions() -> list[dict]:
    base = {
        "dataset_name": "harm_c",
        "image_path": "image.png",
        "ocr_text_full": "Yeah right this symbolic caption mocks America with sarcastic evidence",
        "prob_harmful": 0.8,
        "gold_harmfulness": "harmful",
        "pred_harmfulness": "harmful",
        "target": {"presence": "implicit", "granularity": "community", "attributes": ["nationality"], "label": "community"},
        "gold_target": {"target_presence": "implicit", "target_granularity": "community", "protected_attribute": ["nationality"]},
        "intent": {"primary": "ridicule_mockery", "stance": "hostile", "background_knowledge_needed": True},
        "gold_intent": {"intent_primary": "ridicule_mockery", "stance": "hostile", "background_knowledge_needed": True},
        "tactic": {"rhetorical": ["sarcasm_irony"], "multimodal_relation": "incongruent"},
        "gold_tactic": {"tactic_rhetorical": ["sarcasm_irony"], "tactic_multimodal_relation": "incongruent"},
        "gold_evidence_text": ["sarcastic evidence"],
        "supporting_evidence": {"internal": [{"text": "sarcastic evidence", "score": 0.9}], "external": [{"text": "America context", "score": 0.7}]},
        "stage_metadata": {"stage_a": {"roi_count": 2}, "stage_b": {"retrieved_count": 1}, "stage_c": {"filtered_candidate_count": 1}, "stage_d": {"knowledge_need": 0.8}},
        "rationale": "The rationale mentions sarcastic evidence and America context, matching the harmful ridicule intent.",
    }
    return [
        {"sample_id": "tp", "gold_label": 1, "pred_label": 1, **base},
        {"sample_id": "tn", "gold_label": 0, "pred_label": 0, **base, "gold_harmfulness": "non_harmful", "pred_harmfulness": "non_harmful"},
        {"sample_id": "fp", "gold_label": 0, "pred_label": 1, **base, "gold_harmfulness": "non_harmful"},
        {"sample_id": "fn", "gold_label": 1, "pred_label": 0, **base, "pred_harmfulness": "non_harmful"},
    ]


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
  text:
    prefer_transformers: false
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
    records = [{"id": f"s{idx}", "image": f"s{idx}.png", "labels": idx % 2, "text": f"sample {idx} sarcastic evidence"} for idx in range(4)]
    for record in records:
        (root / "img" / record["image"]).write_bytes(b"not-real")
    for name, subset in {"all": records, "train": records[:2], "val": records[2:3], "test": records[3:]}.items():
        (root / "txt" / f"{name}.jsonl").write_text("\n".join(json.dumps(row) for row in subset) + "\n", encoding="utf-8")
    ann = tmp_path / "annotation" / "harm_c"
    ann.mkdir(parents=True)
    (ann / "harmc_annotations.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "sample_id": row["id"],
                    "annotation": {
                        "target": {"target_presence": "implicit", "target_granularity": "community"},
                        "intent": {"intent_primary": "ridicule_mockery", "background_knowledge_needed": True},
                        "tactic": {"tactic_rhetorical": ["sarcasm_irony"], "tactic_multimodal_relation": "incongruent"},
                    },
                }
            )
            for row in records
        )
        + "\n",
        encoding="utf-8",
    )
    return source


def _write_csv(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
