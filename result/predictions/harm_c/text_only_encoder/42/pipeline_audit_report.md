# Pipeline Audit Report

## Run path
`result/predictions/harm_c/text_only_encoder/42`

## Artifact discovery
- training_log: `result/predictions/harm_c/text_only_encoder/42/training_log.json`
- predictions: `result/predictions/harm_c/text_only_encoder/42/final_predictions.jsonl`
- metrics: `result/predictions/harm_c/text_only_encoder/42/metrics.json`
- manifest: `result/predictions/harm_c/text_only_encoder/42/run_manifest.json`

## Run manifest
- Schema: `experiment_run_manifest_v1`
- Run kind: `baseline`
- Run name: `text_only_encoder`
- Expected knowledge mode: `not_applicable`
- Expected evidence mode: `baseline`

## Ablation contract
- Present: `False`
- Passed: `True`
- Name: `None`

## Training log audit
- Epochs: 1
- Active logits losses: none
- Missing expected logits losses: harmfulness, intent_primary, tactic_multimodal_relation, tactic_rhetorical, target_granularity, target_presence
- Split sizes: `{}`

## Loss provenance summary
- target_presence: `{'provenance': None, 'mean_requires_grad': None, 'provenance_ok': False, 'gradient_ok': False}`
- tactic_multimodal_relation: `{'provenance': None, 'mean_requires_grad': None, 'provenance_ok': False, 'gradient_ok': False}`

## Prediction JSON audit
- Records: 13
- Audited: 5
- Contract passes: 0

## Stage E output provenance
- Stage D trace available: 0/5 audited records

## Evidence attribution provenance
- Internal evidence records: 0
- External evidence records: 0

## Metrics readiness
- Metrics usable: True
- Accuracy: 0.07692307692307693
- Macro-F1: 0.07142857142857144
- Empty split detected: False

## Warnings
- No internal evidence records were present in audited predictions.
- No external evidence records were present in audited predictions.

## Pass/fail summary
**FAIL**

Errors:
- Expected active logits losses are missing: harmfulness, intent_primary, tactic_multimodal_relation, tactic_rhetorical, target_granularity, target_presence.
- target_presence loss provenance is not logits_aux based.
- target_presence mean_requires_grad is not 1.0.
- tactic_multimodal_relation loss provenance is not logits_aux based.
- tactic_multimodal_relation mean_requires_grad is not 1.0.
- Latest training epoch is missing audit fields: active_logits_loss_count, active_logits_losses, active_proxy_loss_count, active_proxy_losses, loss_components, loss_provenance.
- 5/5 audited prediction records violate the Stage E artifact contract. covid_memes_1058: field_provenance.rationale=template, field_provenance.tactic.multimodal_relation=logits_aux, field_provenance.target.presence=logits_aux, output_provenance.cue_fields; covid_memes_1021: field_provenance.rationale=template, field_provenance.tactic.multimodal_relation=logits_aux, field_provenance.target.presence=logits_aux, output_provenance.cue_fields; covid_memes_1025: field_provenance.rationale=template, field_provenance.tactic.multimodal_relation=logits_aux, field_provenance.target.presence=logits_aux, output_provenance.cue_fields
