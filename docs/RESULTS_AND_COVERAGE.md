# Results, Coverage, and Significance

`python scripts/run.py research aggregate` scans only canonical run directories and writes:

- `result/aggregates/all_results.csv`
- `result/aggregates/all_results.json`
- `result/aggregates/main_results.csv`
- `result/aggregates/structured_results.csv`
- `result/aggregates/ablation_results.csv`
- `result/aggregates/knowledge_results.csv`
- `result/aggregates/runtime_results.csv`
- `result/aggregates/significance_results.csv`
- `result/aggregates/coverage_results.csv`
- `result/aggregates/experiment_status.csv`

Long rows identify experiment, family, group, condition, domain role, annotation provenance, seed, task, metric, value, valid-N, total-N, coverage, unknown/ambiguous/masked counts, class distribution, hashes, runtime, and status.

Harmfulness metrics use original binary labels. Structured metrics are mask-aware. `ambiguous`, `unknown`, missing, and clean-ineligible fields are counted separately and excluded according to the label vocabulary. Formal tactic rhetorical metrics use logits-only decoding with a HarMeme-validation-selected threshold.

Mean/std rows include completed, audited numeric results only. Paired significance is eligible only when at least two identical seeds completed for both methods. Missing, blocked, and unavailable values remain blank/`--`; they are never coerced to zero.
