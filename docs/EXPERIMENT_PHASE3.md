# Experiment Phase 3: Analysis and Paper Support

This phase adds analysis scripts for robustness, verifier behavior, difficult subsets, evidence attribution, rationale quality, runtime/cost, statistical significance, and paper-table export. These scripts preserve the Phase 1/2 prediction formats and write new artifacts under `result/`.

## Scripts

```bash
python scripts/run.py analysis cross-domain --setting mixed_train --model ours_full --seed 42
python scripts/run.py analysis cross-domain --setting leave_one_domain_out --heldout harm_c --model ours_full --seed 42
python scripts/run.py analysis cross-domain --setting train_one_test_others --train-dataset facebook --model ours_full --seed 42

python scripts/run.py analysis verifier --dataset harm_c --seed 42
python scripts/run.py analysis subset --dataset all --model ours_full --seed 42
python scripts/run.py analysis select-error-cases --dataset all --model ours_full --seed 42
python scripts/run.py report export-case-data --dataset all --model ours_full --seed 42
python scripts/run.py analysis rationale --dataset all --model ours_full --seed 42
python scripts/run.py analysis runtime --dataset harm_c --limit 200
python scripts/run.py analysis significance
python scripts/run.py report export-paper-tables
```

## Outputs

- `result/predictions_cross_domain/{setting}/{train_or_heldout}/{test_dataset}/{model}/{seed}/final_predictions.jsonl`
- `result/metrics/cross_domain.csv`
- `result/metrics/verifier_eval.csv`
- `result/analysis/verifier_examples.jsonl`
- `result/metrics/subset_analysis.csv`
- `result/analysis/error_cases/error_cases.jsonl`
- `result/analysis/error_cases/{false_positive,false_negative,true_positive,true_negative}/*.json`
- `result/analysis/error_cases/case_visualization_data.jsonl`
- `result/metrics/rationale_eval.csv`
- `result/analysis/rationale_human_eval_template.csv`
- `result/metrics/runtime_cost.csv`
- `result/analysis/runtime_cost_details.json`
- `result/metrics/significance_tests.csv`
- `result/paper_tables/table1_dataset_statistics.csv` through `table8_runtime_cost.csv`

## Weak-Label Assumptions

Verifier evaluation can run without human verifier labels. In that case, relevance and support labels are weakly inferred from overlap between candidate knowledge and annotation evidence text / structured labels. Rationale evaluation uses automatic proxies such as evidence overlap, named-entity hallucination, and label consistency; the exported CSV template is intended for later human scoring.

## Notes

Cross-domain runs currently support fresh execution of `ours_full`. The default `--epochs 0` provides a fast robustness pass over the existing pipeline; set `--epochs` above zero to train lightweight projection/reasoning/head parameters for a domain setting.
