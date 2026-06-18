# Framework Implementation Summary

This document summarizes the modular harmful meme interpretation framework code,
the responsibility of each implementation folder, and the outputs generated when
the framework is executed.

## High-Level Layout

```text
meme_detection/
├── configs/
├── dataset/
├── module/
│   ├── backbones/
│   ├── losses/
│   ├── pipeline/
│   ├── stage_a/
│   ├── stage_b/
│   ├── stage_c/
│   ├── stage_d/
│   └── stage_e/
├── scripts/
├── utils/
├── tests/
├── docs/
└── result/
```

The implementation follows a five-stage pipeline:

1. Stage A: Internal Evidence Extractor
2. Stage B: External Knowledge Acquisition
3. Stage C: Knowledge Relevance Filter / Verifier
4. Stage D: Evidence Fusion & Reasoning
5. Stage E: Structured Interpretation Head

## `configs/`

Configuration files control paths, model dimensions, optional backbones, stage
settings, retrieval options, and output paths.

- `default.yaml`: main runtime configuration.
- `pipeline.yaml`: full pipeline configuration.
- `stage_a.yaml` to `stage_e.yaml`: stage-specific configuration examples.

Important settings include:

- dataset root: `dataset/V1`
- annotation root: `outputs`
- result root: `result`
- hidden dimension: `256`
- retrieval corpus: `dataset/V1/wiki_common/wiki_corpus.jsonl`
- fallback-safe defaults for CLIP and HuggingFace encoders

## `dataset/`

The dataset package provides unified meme sample loading.

- `base_dataset.py`: defines the `MemeSample` dataclass and common dataset
  preview/validation utilities.
- `meme_dataset.py`: scans `dataset/V1/*/img` and `dataset/V1/*/txt`,
  pairs images with OCR text, maps source folders to logical dataset names, and
  joins annotation JSONL records.
- `collate.py`: provides `meme_collate_fn` for PyTorch `DataLoader` usage.

Logical dataset mapping:

```text
covid_img+text     -> harm_c
political_img+text -> harm_p
memotion_img+text  -> memotion
facebook_img+text  -> facebook
```

Each loaded sample contains:

- `sample_id`
- `dataset_name`
- `image_path`
- `ocr_text_full`
- `raw_label`
- `annotation`
- `raw_record`
- `metadata`

## `module/backbones/`

Backbone wrappers isolate optional heavy dependencies from the stage logic.

- `clip_wrapper.py`: image encoder wrapper. Uses OpenCLIP/CLIP if configured
  and available; otherwise uses deterministic image fallback features.
- `text_encoder_wrapper.py`: text encoder wrapper. Uses local HuggingFace
  models if configured and available; otherwise uses hashed token embeddings.
- `detector_adapter.py`: local object/symbol detection interface with heuristic
  pseudo-ROI fallback.
- `retriever_adapter.py`: local sparse+dense retrieval adapter over JSONL/text
  corpora with fallback candidates.
- `cross_encoder_adapter.py`: lightweight pairwise relevance/support scoring.
- `generator_adapter.py`: deterministic template hypothesis generation.

These wrappers keep the framework runnable offline while preserving extension
points for stronger pretrained models.

## `module/stage_a/`

Stage A extracts internal evidence from the meme image and OCR text.

Key files:

- `schemas.py`: dataclass inputs/outputs for Stage A.
- `visual_encoder.py`: global visual and coarse patch token extraction.
- `text_semantic_encoder.py`: global and token-level OCR text encoding.
- `local_symbol_extractor.py`: ROI/object/symbol evidence extraction.
- `incongruity_analyzer.py`: attention-based visual-text relation and
  incongruity scoring.
- `internal_evidence_extractor.py`: orchestrates Stage A.

Stage A output includes:

- `internal_tokens`: internal evidence token bank, shape `[N_int, 256]`
- `evidence_items`: text/visual/ROI/incongruity evidence metadata
- `global_visual`
- `global_text`
- `patch_tokens`
- `text_tokens`
- `roi_tokens`
- auxiliary scores such as:
  - `knowledge_need`
  - `multimodal_relation`
  - `target_presence`
  - `rhetorical_cues`

## `module/stage_b/`

Stage B acquires external knowledge.

Key files:

- `schemas.py`: query, linked entity, candidate, and output dataclasses.
- `query_constructor.py`: builds OCR, entity, event, meme-template,
  social-context, and target-hypothesis queries.
- `entity_linker.py`: lightweight entity/concept linking with alias expansion.
- `hybrid_retriever.py`: sparse+dense retrieval, score fusion, and reranking.
- `context_generator.py`: generates hypotheses grounded in retrieved evidence.
- `external_knowledge_acquisition.py`: orchestrates Stage B.

Stage B output includes:

- `query_bundle`
- `linked_entities`
- `knowledge_candidates`
- `candidate_tokens`, shape `[M, 256]`
- generated context hypotheses
- retrieval metadata and query-type counts

## `module/stage_c/`

Stage C filters and verifies external knowledge.

Key files:

- `schemas.py`: verified knowledge dataclasses.
- `relevance_scorer.py`: evidence-aware pair scoring using
  `[q; k; q*k; |q-k|]` features.
- `support_verifier.py`: target/intent/tactic support classification.
- `validator.py`: source credibility, temporal, cultural, and language checks.
- `redundancy_reducer.py`: diversity-aware knowledge selection.
- `knowledge_filter_verifier.py`: orchestrates Stage C.

Stage C output includes:

- `verified_items`
- `verified_tokens`, shape `[K, 256]`
- `support_matrix`
- `final_scores`
- `internal_summary`
- score metadata for relevance, support, validity, and redundancy

## `module/stage_d/`

Stage D fuses internal evidence with verified knowledge.

Key files:

- `schemas.py`: fusion output dataclasses.
- `internal_aggregator.py`: multi-token internal memory encoder.
- `knowledge_cross_attention.py`: knowledge-conditioned cross-attention.
- `gating.py`: token-level, sample-level, task-level, and head-level gates.
- `task_reasoning.py`: shared and task-private reasoning latents.
- `evidence_fusion_reasoning.py`: orchestrates Stage D.

Stage D output includes:

- `shared_reasoning_state`, shape `[256]`
- `internal_memory`, shape `[N_int, 256]`
- `fused_tokens`, shape `[N_int, 256]`
- `cross_attention_weights`
- gates for evidence attribution and ablation
- task latents for:
  - target
  - intent
  - tactic
- regularizer hooks such as token sparsity and knowledge sufficiency

## `module/stage_e/`

Stage E produces final structured interpretation.

Key files:

- `schemas.py`: prediction and final output dataclasses.
- `harmfulness_head.py`: harmful/non-harmful prediction.
- `target_head.py`: target granularity prediction.
- `intent_head.py`: intent prediction.
- `tactic_head.py`: rhetorical/multimodal tactic prediction.
- `evidence_attribution.py`: internal/external evidence selectors.
- `rationale_generator.py`: template rationale generator with optional
  constrained generator hook.
- `structured_interpretation_head.py`: orchestrates Stage E.

Stage E output follows this structure:

```json
{
  "harmfulness": {...},
  "target": {...},
  "intent": {...},
  "tactic": {...},
  "supporting_evidence": {
    "internal": [...],
    "external": [...]
  },
  "rationale": "..."
}
```

The structured output also includes training hooks with score distributions and
Stage D regularizer metadata.

## `module/pipeline/`

The pipeline package wires all stages together.

- `model.py`: `HarmfulMemePipeline`, the top-level model that runs stages A-E.
- `runner.py`: `PipelineRunner`, which loads data, executes the pipeline, and
  saves results.
- `inference.py`: helper for running a single in-memory sample.

The pipeline supports running through a selected stage using `run_until`:

```text
a, b, c, d, e
```

## `module/losses/`

Reusable research loss utilities.

- `structured_losses.py`: includes classification losses, evidence attribution
  loss, consistency regularization, and aggregate structured loss computation.

Supported supervision hooks include:

- harmfulness loss
- target loss
- intent loss
- tactic loss
- evidence attribution loss
- compatibility/consistency loss
- optional rationale/auxiliary loss hooks

## `utils/`

Shared utility code.

- `io.py`: YAML/JSON/JSONL loading and saving.
- `logging_utils.py`: logger setup.
- `image_utils.py`: image loading and fallback image features.
- `text_utils.py`: normalization, tokenization, keyword extraction, rhetorical
  cue detection, target-presence heuristics.
- `tensor_utils.py`: tensor serialization and deterministic hashed vectors.
- `retrieval_utils.py`: lexical scoring, BM25-like scoring, rank fusion.
- `eval_utils.py`: binary metrics, multiclass summaries, evidence
  precision/recall.
- `seed.py`: reproducibility helper.

## `scripts/`

Command-line entry points.

- `inspect_dataset.py`: prints dataset statistics and sample previews.
- `run_stage_a.py`: runs Stage A and optionally saves internal evidence.
- `run_stage_b.py`: runs Stages A-B and optionally saves knowledge candidates.
- `run_stage_c.py`: runs Stages A-C and optionally saves verified knowledge.
- `run_stage_d.py`: runs Stages A-D and optionally saves fusion states.
- `run_stage_e.py`: runs the full A-E stack.
- `run_pipeline.py`: configurable full pipeline runner.
- `export_intermediate_results.py`: writes a manifest of result files.

Example commands:

```bash
python scripts/inspect_dataset.py --limit 5
python scripts/run_stage_a.py --dataset harm_c --limit 2
python scripts/run_stage_b.py --dataset harm_c --limit 2
python scripts/run_pipeline.py --dataset harm_c --limit 2
python scripts/export_intermediate_results.py
```

## `tests/`

Smoke tests cover:

- dataset loading
- each stage forward pass
- full pipeline execution
- Phase 2 loss and evaluation utilities

Run tests with:

```bash
python -m pytest tests -q
```

## Generated Outputs

When the pipeline is executed with saving enabled, outputs are written under
`result/`.

### Dataset Inspection

Command:

```bash
python scripts/inspect_dataset.py --limit 5
```

Output:

- Printed JSON summary to stdout.
- Includes dataset statistics, validation counts, label distribution, and sample
  previews.

### Stage A Output

Path:

```text
result/stage_a/internal_evidence.jsonl
```

Contains one JSONL record per sample with:

- internal evidence token metadata
- global visual/text evidence
- text-span evidence
- visual-patch evidence
- ROI/local-symbol evidence
- cross-modal incongruity evidence
- auxiliary scores and tensor-shape metadata

Tensor values are compacted for readable JSONL output.

### Stage B Output

Path:

```text
result/stage_b/knowledge_candidates.jsonl
```

Contains:

- query bundle
- linked entities/concepts
- retrieved knowledge candidates
- generated hypotheses
- retrieval scores
- sparse/dense/fusion/reranking metadata
- candidate token shape previews

### Stage C Output

Path:

```text
result/stage_c/verified_knowledge.jsonl
```

Contains:

- selected verified knowledge items
- relevance, support, validity, and final scores
- claim-level target/intent/tactic support labels
- redundancy cluster metadata
- verified token previews
- support matrix previews

### Stage D Output

Paths:

```text
result/stage_d/fusion_states.pt
result/stage_d/fusion_outputs.pt
```

`fusion_states.pt` is the current tensor artifact. `fusion_outputs.pt` is kept
as a backward-compatible alias.

Contains PyTorch tensors for:

- shared reasoning state
- internal memory
- fused tokens
- cross-attention weights
- task latents
- gates
- regularizer metadata

### Stage E Output

Path:

```text
result/stage_e/final_predictions.jsonl
```

Contains final structured predictions:

- harmfulness label and score distribution
- target presence, granularity, attributes, and summary
- intent primary/secondary labels, stance, and background-knowledge need
- tactic labels, multimodal relation, and structural tactic labels
- selected internal/external evidence
- evidence-grounded rationale
- training hooks

### Analysis Exports

Paths:

```text
result/analysis/evidence_attribution.jsonl
result/analysis/sample_summaries.jsonl
```

`evidence_attribution.jsonl` contains only the selected internal and external
evidence for each sample.

`sample_summaries.jsonl` contains compact per-sample prediction summaries useful
for manual inspection.

### Intermediate Manifest

Command:

```bash
python scripts/export_intermediate_results.py
```

Output:

```text
result/intermediate_manifest.json
```

Contains a list of generated result files and their file sizes.

## Fallback Behavior

The framework remains runnable without optional heavy dependencies:

- no CLIP/OpenCLIP installed: deterministic fallback image features are used
- no local HuggingFace model: hashed text embeddings are used
- no retrieval corpus: fallback knowledge candidates are generated
- no object detector: heuristic pseudo-ROIs are used
- no constrained generator: template rationale generation is used

This keeps local smoke tests and CLI runs stable while preserving extension
points for stronger Phase 3 training and model integration.
