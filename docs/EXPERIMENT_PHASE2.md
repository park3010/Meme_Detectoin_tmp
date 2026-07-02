# Experiment Phase 2: Full Framework Experiments

This phase adds paper-facing experiment wrappers around the existing 5-stage harmful meme interpretation pipeline. The core framework modules remain unchanged; these utilities run variants of `HarmfulMemePipeline`, save predictions under `result/predictions`, and aggregate metrics under `result/metrics`.

## Added Experiment Code

- `experiments/train_ours.py`: trains/evaluates Ours Full with heavy backbones frozen by default and lightweight Stage D/E heads trainable.
- `experiments/ablation_configs.py`: names supported stage-wise ablations, knowledge modes, and fusion modes.
- `experiments/ablation_runner.py`: runs ablation and fusion variants by transforming stage outputs around the existing pipeline.
- `experiments/knowledge_comparison.py`: runs no/generated/retrieved/verified knowledge modes and exports sample-level knowledge analysis.
- `experiments/structured_eval.py`: evaluates harmfulness plus target, intent, tactic, and weak evidence matching metrics.
- `experiments/prediction_io.py`: serializes framework outputs into compact JSONL prediction rows.

## Scripts

```bash
python scripts/run.py train --experiment ours_full --dataset harm_c --seed 42 --epochs 5
python scripts/evaluate_structured_outputs.py --dataset harm_c --model ours_full --seed 42

python scripts/run.py ablation --dataset harm_c --ablation w_o_roi --seed 42
python scripts/run.py ablation --dataset harm_c --ablation w_o_support_verifier --seed 42
python scripts/run.py ablation --dataset harm_c --fusion-mode task_aware_gate_verified --seed 42

python scripts/run_knowledge_comparison.py --dataset harm_c --mode no_knowledge --seed 42
python scripts/run_knowledge_comparison.py --dataset harm_c --mode verified --seed 42

python scripts/aggregate_structured_results.py
```

## Supported Modes

Ablations: `full`, `w_o_roi`, `w_o_incongruity`, `w_o_retrieval`, `w_o_context_generation`, `w_o_relevance_scorer`, `w_o_support_verifier`, `w_o_temporal_cultural_validator`, `w_o_task_aware_gate`, `w_o_structured_auxiliary`, `label_only_no_evidence`. The CLI also accepts `w_o_verifier` as an alias for `w_o_support_verifier`.

Knowledge modes: `no_knowledge`, `generated_only`, `retrieved_only`, `generated_retrieved`, `verified`.

Fusion modes: `concat_mlp`, `mean_pooling`, `cross_attention`, `shared_gate`, `task_aware_gate`, `task_aware_gate_verified`.

## Outputs

- `result/predictions/{dataset}/ours_full/{seed}/final_predictions.jsonl`
- `result/predictions/{dataset}/ablation_{name}/{seed}/final_predictions.jsonl`
- `result/predictions/{dataset}/knowledge_{mode}/{seed}/final_predictions.jsonl`
- `result/metrics/structured_interpretation.csv`
- `result/metrics/ablation.csv`
- `result/metrics/knowledge_comparison.csv`
- `result/metrics/fusion_comparison.csv`
- `result/analysis/knowledge_comparison_examples.jsonl`
- `result/analysis/gate_summary.jsonl`

## Notes

The ablation and knowledge runners are experiment-level wrappers: they execute the existing Stage A-E modules and alter intermediate outputs to simulate controlled removals. Ours Full includes a lightweight training loop; ablation variants currently emphasize reproducible evaluation and export infrastructure for the first experiment pass.
