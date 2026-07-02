# Phase 2 Upgrade Notes

This phase upgrades the Phase 1 scaffold while preserving the stage interfaces
and CLI entry points.

## Main Changes

- Stage A now builds a richer internal evidence bank with global visual tokens,
  text-span tokens, visual-patch tokens, ROI tokens, and an attention-based
  cross-modal incongruity token.
- Stage B now separates sparse and dense local retrieval, fuses rankings,
  supports JSONL/text corpora, expands entity aliases, and generates hypotheses
  from retrieved evidence.
- Stage C now scores `[q; k; q*k; |q-k|]` pairwise features, verifies
  target/intent/tactic claims separately, validates source/time/culture, and
  performs diversity-aware knowledge selection.
- Stage D now uses support-aware cross-attention, token/sample/head gates,
  task-private reasoning pools, and regularizer hook scalars.
- Stage E now emits richer structured JSON for harmfulness, target, intent,
  tactic, supporting evidence, rationale, and training hooks.
- Research utilities were added for structured losses and evaluation metrics.
- Result exports now include `result/stage_d/fusion_states.pt`,
  `result/analysis/evidence_attribution.jsonl`, and
  `result/analysis/sample_summaries.jsonl`.

## Running

```bash
python scripts/inspect_dataset.py --limit 5
python scripts/run.py stage --dataset harm_c --until stage_b --limit 2
python scripts/run.py stage --dataset harm_c --until stage_e --limit 2
python scripts/export_intermediate_results.py
```

## Current Limits

- CLIP/OpenCLIP and HuggingFace encoders are still optional and default off for
  offline-safe runs.
- Dense retrieval uses deterministic hashed vectors as a FAISS-compatible
  fallback, not a trained bi-encoder index.
- Structured heads are research-ready modules with heuristic priors, not trained
  classifiers yet.
- Rationale generation remains template-based unless a constrained generator is
  injected.
