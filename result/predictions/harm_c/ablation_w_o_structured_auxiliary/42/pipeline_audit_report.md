# Pipeline Audit Report

## Run path
`result/predictions/harm_c/ablation_w_o_structured_auxiliary/42`

## Artifact discovery
- training_log: `result/predictions/harm_c/ablation_w_o_structured_auxiliary/42/training_log.json`
- predictions: `result/predictions/harm_c/ablation_w_o_structured_auxiliary/42/final_predictions.jsonl`
- metrics: `result/predictions/harm_c/ablation_w_o_structured_auxiliary/42/metrics.json`
- manifest: `result/predictions/harm_c/ablation_w_o_structured_auxiliary/42/run_manifest.json`

## Run manifest
- Schema: `experiment_run_manifest_v1`
- Run kind: `ablation`
- Run name: `ablation_w_o_structured_auxiliary`
- Expected knowledge mode: `verified`
- Expected evidence mode: `internal_external_evidence`

## Ablation contract
- Present: `True`
- Passed: `True`
- Name: `w_o_structured_auxiliary`

## Training log audit
- Epochs: 1
- Active logits losses: harmfulness, intent_primary, tactic_rhetorical, target_granularity
- Missing expected logits losses: none
- Split sizes: `{'train': 84, 'valid': 3, 'test': 13}`

## Loss provenance summary
- target_presence: `{'disabled_by_contract': True}`
- tactic_multimodal_relation: `{'disabled_by_contract': True}`

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
