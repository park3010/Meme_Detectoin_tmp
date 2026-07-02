# Pipeline Audit Report

## Run path
`result/predictions/harm_c/ablation_w_o_retrieval/42`

## Artifact discovery
- training_log: missing
- predictions: `result/predictions/harm_c/ablation_w_o_retrieval/42/final_predictions.jsonl`
- metrics: `result/predictions/harm_c/ablation_w_o_retrieval/42/metrics.json`
- manifest: `result/predictions/harm_c/ablation_w_o_retrieval/42/run_manifest.json`

## Run manifest
- Schema: `experiment_run_manifest_v1`
- Run kind: `ablation`
- Run name: `ablation_w_o_retrieval`
- Expected knowledge mode: `no_knowledge`
- Expected evidence mode: `internal_only`

## Ablation contract
- Present: `True`
- Passed: `True`
- Name: `w_o_retrieval`

## Training log audit
- Epochs: 0
- Active logits losses: none
- Missing expected logits losses: none
- Split sizes: `{}`

## Loss provenance summary
- target_presence: `{}`
- tactic_multimodal_relation: `{}`

## Prediction JSON audit
- Records: 13
- Audited: 5
- Contract passes: 5

## Stage E output provenance
- Stage D trace available: 5/5 audited records

## Evidence attribution provenance
- Internal evidence records: 15
- External evidence records: 0

## Metrics readiness
- Metrics usable: True
- Accuracy: 0.9230769230769231
- Macro-F1: 0.48000000000000004
- Empty split detected: False

## Warnings
- Training log artifact was not found.
- Training log is absent or empty.
- No external evidence records were present in audited predictions.

## Pass/fail summary
**WARNING**
