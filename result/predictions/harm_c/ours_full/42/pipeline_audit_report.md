# Pipeline Audit Report

## Run path
`result/predictions/harm_c/ours_full/42`

## Artifact discovery
- training_log: `result/predictions/harm_c/ours_full/42/training_log.json`
- predictions: `result/predictions/harm_c/ours_full/42/final_predictions.jsonl`
- metrics: `result/predictions/harm_c/ours_full/42/metrics.json`
- manifest: `result/predictions/harm_c/ours_full/42/run_manifest.json`

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
- Split sizes: `{'train': 84, 'valid': 3, 'test': 13}`

## Loss provenance summary
- target_presence: `{'provenance': 'logits_aux_with_proxy_fallback', 'mean_requires_grad': 1.0, 'provenance_ok': True, 'gradient_ok': True}`
- tactic_multimodal_relation: `{'provenance': 'logits_aux_with_proxy_fallback', 'mean_requires_grad': 1.0, 'provenance_ok': True, 'gradient_ok': True}`

## Prediction JSON audit
- Records: 13
- Audited: 5
- Contract passes: 5

## Stage E output provenance
- Stage D trace available: 5/5 audited records

## Evidence attribution provenance
- Internal evidence records: 15
- External evidence records: 15

## Metrics readiness
- Metrics usable: True
- Accuracy: 0.9230769230769231
- Macro-F1: 0.48000000000000004
- Empty split detected: False

## Warnings
- None

## Pass/fail summary
**PASS**
