# Research Refinement Notes

This note documents the latest refinement pass on the modular harmful meme
interpretation framework. The changes preserve the existing five-stage
architecture and CLI behavior.

## Stage B: Evidence-Aware Retrieval Signals

Stage B now augments OCR-only linking with Stage A evidence signals:

- full OCR text
- `text_span` evidence
- `local_symbol` evidence
- evidence metadata keywords
- ROI/local-symbol labels
- multimodal relation labels
- rhetorical cue labels

These surfaces are used for entity/concept linking and query augmentation.
Additional visual-symbol-aware, meme-template, social-context, and target-span
queries are added when Stage A evidence suggests them.

Candidate knowledge tokens remain `[M, 256]`, but now combine:

- candidate text embedding
- source-type embedding
- query-type embedding
- score feature projection

Stage B metadata now reports evidence surface counts, visual evidence usage,
fallback candidate usage, and query source breakdowns.

## Stage C: Support Matrix Semantics

Stage C keeps the backward-compatible 6-column `support_matrix`:

```text
[relevance, target_support, intent_support, tactic_support, validity, final]
```

The column names are now stored in metadata as `support_matrix_columns`, and
`StageCOutput.task_support_matrix` exposes columns `1:4` for downstream
target/intent/tactic reasoning.

Verified knowledge metadata now includes:

- claim-level support for target, intent, and tactic
- support labels per claim
- relevance components
- validity components
- final score components
- redundancy/cluster metadata

Fallback knowledge can still pass low-relevance filtering for smoke tests by
default. Set `allow_low_relevance_fallback: false` in Stage C config or module
construction to apply `min_relevance` to fallback candidates too.

## Stage D: Gated Evidence Fusion

Stage D now passes Stage C task-support information into knowledge-conditioned
cross-attention. The gate compares internal and fused memory and performs
token-level gated fusion:

```text
gated = gate * fused_memory + (1 - gate) * internal_memory
```

Sample-level gating also uses Stage A `knowledge_need` when available, reducing
external knowledge influence when the meme appears interpretable from internal
evidence alone.

Stage D metadata now records:

- `knowledge_need`
- `support_matrix_shape`
- `gate_mode`
- whether task support was used

## Stage E: Schema-Aligned Structured Outputs

Stage E keeps the existing top-level keys:

- `harmfulness`
- `target`
- `intent`
- `tactic`
- `supporting_evidence`
- `rationale`
- `training_hooks`

The nested payloads are more annotation-compatible:

- target: `presence`, `granularity`, `attributes`, `label`, `label_summary`,
  `score`, `scores`
- intent: `primary`, `stance`, `secondary`, `background_knowledge_needed`,
  `background_knowledge_score`, `score`, `scores`
- tactic: `rhetorical`, `multimodal_relation`, `structural`, `score`,
  `scores`, `keywords`

Prediction objects now optionally carry trainable logits and label spaces while
still exposing detached score dictionaries for JSON export and smoke tests.

## Structured Losses

`module/losses/structured_losses.py` now supports both:

- logits-based losses for gradient-based training
- score-dictionary losses for detached analysis and smoke tests

Added helpers include:

- `classification_loss_from_logits`
- `multilabel_loss_from_logits`
- `binary_loss_from_logits`
- `extract_supervision_from_annotation`

The annotation parser flattens the nested annotation schema into optional
supervision fields such as:

- `harmfulness`
- `target_presence`
- `target_granularity`
- `target_attributes`
- `intent_primary`
- `stance`
- `secondary_intent`
- `background_knowledge_needed`
- `tactic_rhetorical`
- `tactic_multimodal_relation`
- `evidence_text`

All losses are optional and skipped cleanly when supervision is missing.
