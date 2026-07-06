# Pipeline Audit Report

## Run path
`result_cleanup_cpu_smoke_no_tqdm_limit20/predictions/harm_c/ours_full/42`

## Artifact discovery
- training_log: `result_cleanup_cpu_smoke_no_tqdm_limit20/predictions/harm_c/ours_full/42/training_log.json`
- predictions: `result_cleanup_cpu_smoke_no_tqdm_limit20/predictions/harm_c/ours_full/42/final_predictions.jsonl`
- metrics: `result_cleanup_cpu_smoke_no_tqdm_limit20/predictions/harm_c/ours_full/42/metrics.json`
- manifest: `result_cleanup_cpu_smoke_no_tqdm_limit20/predictions/harm_c/ours_full/42/run_manifest.json`
- tactic_decoding: `result_cleanup_cpu_smoke_no_tqdm_limit20/predictions/harm_c/ours_full/42/tactic_rhetorical_decoding.json`
- best_model: `result_cleanup_cpu_smoke_no_tqdm_limit20/predictions/harm_c/ours_full/42/best_model.pt`
- validation_predictions: `result_cleanup_cpu_smoke_no_tqdm_limit20/predictions/harm_c/ours_full/42/validation_predictions.jsonl`

## Run manifest
- Schema: `experiment_run_manifest_v1`
- Run kind: `ours_full`
- Run name: `ours_full`
- Expected knowledge mode: `verified`
- Expected evidence mode: `internal_external_evidence`

## Ablation contract
- Present: `False`
- Passed: `True`
- Name: `None`

## Training log audit
- Epochs: 1
- Active logits losses: harmfulness, intent_primary, tactic_multimodal_relation, tactic_rhetorical, target_granularity, target_presence
- Missing expected logits losses: none
- Split sizes: `{'train': 19, 'valid': 1, 'test': 0}`

## Loss provenance summary
- target_presence: `{'provenance': 'logits_aux_with_proxy_fallback', 'mean_requires_grad': 1.0, 'provenance_ok': True, 'gradient_ok': True}`
- tactic_multimodal_relation: `{'provenance': 'logits_aux_with_proxy_fallback', 'mean_requires_grad': 1.0, 'provenance_ok': True, 'gradient_ok': True}`

## Prediction JSON audit
- Records: 0
- Audited: 0
- Contract passes: 0

## Stage E output provenance
- Stage D trace available: 0/0 audited records

## Evidence attribution provenance
- Internal evidence records: 0
- External evidence records: 0

## Metrics readiness
- Metrics usable: True
- Accuracy: None
- Macro-F1: None
- Empty split detected: True

## Formal tactic decoding
- Required: True
- Artifact found: True
- Passed: False
- Source: `tactic_logits_sigmoid`
- Selected threshold: `0.5`
- Formal metric status: `blocked`

## Warnings
- None

## Pass/fail summary
**FAIL**

Errors:
- Prediction artifact is absent or contains zero records.
- Formal tactic decoding audit failed: metrics.tactic_rhetorical_formal_status_ready.
- Formal tactic decoding audit failed: metrics.tactic_rhetorical_eligible_sample_count_nonzero.
