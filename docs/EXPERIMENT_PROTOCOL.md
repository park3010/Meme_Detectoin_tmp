# Experiment Protocol Lock

This repository uses `scripts/run.py suite` as the canonical entry point for reproducible paper experiment batches. The suite runner resolves a preset from `configs/config.yaml`, pins one split file per dataset/seed, executes the existing training or evaluation functions, writes run manifests, and optionally audits each full-framework or ablation run.

## Experiment 0 Preflight

Before running `core_1seed` or `core_5seed`, run Experiment 0:

```bash
python scripts/run.py preflight \
  --profile smoke \
  --config configs/config.yaml \
  --dataset harm_c harm_p facebook memotion \
  --seed 42 \
  --label-set clean \
  --device cpu \
  --write-report
```

Then run the strict main-experiment gate:

```bash
python scripts/run.py preflight \
  --profile main_experiment \
  --config configs/config.yaml \
  --dataset harm_c harm_p facebook memotion \
  --seed 42 \
  --label-set clean \
  --device cpu \
  --write-report \
  --strict
```

Smoke preflight is an offline structural check and may pass with fallback encoders. Main-experiment strict preflight is the paper-readiness gate and must pass before main experiment results should be treated as valid.

Backend availability is not the same thing as pretrained weight loading. The preflight records actual `weights_loaded`, `weights_source`, `fallback_used`, and `random_initialization_used` states for vision and text backbones. A random OpenCLIP model or hashing text encoder blocks `main_experiment`.

The preflight also verifies:

- dataset-field eligibility using normalized labels and `LabelVocab` ignore policies
- fixed split integrity and split SHA-256 reuse
- retrieval corpus parseability and provenance
- metric-contract readiness, including the rule that formal `tactic_rhetorical` metrics must use logits-only decoding rather than rendered heuristic labels
- normalized annotation snapshot hashes

Preflight artifacts are written under:

```text
result/preflight/<profile>/
```

Decision meanings:

- `PASS`: no warnings or blocking errors
- `PASS_WITH_WARNINGS`: runnable but not paper-ready without reviewing warnings
- `BLOCKED`: do not run main experiments until the blocking errors are resolved

## Suite Presets

Configured under `experiments.suites` in `configs/config.yaml`:

- `core_smoke`: one small harm_c validation run with Ours Full, text baseline, and key ablations.
- `core_1seed`: one-seed main comparison across all datasets.
- `core_5seed`: five-seed main comparison across all datasets.
- `knowledge_1seed`: knowledge-source comparison for selected datasets.

Dry-run a suite without writing artifacts:

```bash
python scripts/run.py suite --suite core_smoke --config configs/config.yaml --device cpu --dry-run
```

Run the smoke protocol with audits:

```bash
python scripts/run.py suite \
  --suite core_smoke \
  --config configs/config.yaml \
  --device cpu \
  --disable-tqdm \
  --audit-after-run \
  --strict \
  --require-nonempty-metrics
```

## Run Manifests

Every suite full-model, baseline, ablation, fusion, and knowledge-comparison run writes:

```text
result/predictions/<dataset>/<run_name>/<seed>/run_manifest.json
```

The manifest uses schema `experiment_run_manifest_v1` and records:

- suite name, run kind, run name, dataset, seed
- config path and SHA-256
- split file path and SHA-256
- requested command
- ablation contract, when applicable
- component-state flags
- expected active logits losses and intentionally disabled losses
- expected knowledge/evidence mode

Suite-level tracking is written to:

```text
result/experiment_suites/<suite_name>/suite_manifest.json
```

## Formal `tactic_rhetorical` Metric

The paper-facing rhetorical tactic metric is logits-only.

- Source: trainable `TacticHead` tactic logits serialized as `tactic_rhetorical_logits`.
- Probability: sigmoid over the trainable logits.
- Threshold: one global threshold selected on validation macro-F1 for each dataset/run/seed/best checkpoint.
- Test: the selected validation threshold is applied unchanged to test predictions.
- Label universe: non-ignored `tactic_rhetorical` labels from the canonical vocab / head label order.
- `none`: fallback only when no non-none label reaches threshold; it is never independently thresholded.
- Rendered fields excluded: `tactic.rhetorical`, `tactic.rhetorical_labels`, heuristic rhetorical cues, Stage A cues, rationale text, and top-1-plus-heuristic rendering.

The historical rendered tactic metrics remain legacy diagnostics. They are useful for explanation-output sanity checks, but they are not paper-facing performance metrics.

Training finalization writes:

```text
result/predictions/<dataset>/<run_name>/<seed>/validation_predictions.jsonl
result/predictions/<dataset>/<run_name>/<seed>/final_predictions.jsonl
result/predictions/<dataset>/<run_name>/<seed>/tactic_rhetorical_decoding.json
```

The validation and test prediction files keep the existing `tactic` payload unchanged and add an evaluation-only trace:

```text
evaluation.tactic_rhetorical_formal
```

The formal metrics are named with an explicit logits-only suffix:

- `tactic_rhetorical_macro_f1_logits_only`
- `tactic_rhetorical_micro_f1_logits_only`
- `tactic_rhetorical_none_f1`
- `tactic_rhetorical_exact_match_ratio`
- `tactic_rhetorical_eligible_sample_count`
- `tactic_rhetorical_validation_selected_threshold`
- `tactic_rhetorical_validation_macro_f1_at_selected_threshold`

Standalone evaluation can reproduce the formal metric from saved predictions:

```bash
python scripts/run.py evaluate \
  --dataset harm_c \
  --validation-predictions result/predictions/harm_c/ours_full/42/validation_predictions.jsonl \
  --test-predictions result/predictions/harm_c/ours_full/42/final_predictions.jsonl \
  --formal-tactic-metrics \
  --config configs/config.yaml \
  --label-set clean
```

By default, standalone evaluation writes `tactic_rhetorical_decoding_eval.json` beside the test prediction file. Pass `--decoding-artifact` only when you intentionally want a different path.

## Ablation Contracts

Ablation semantics are declared in `experiments/ablation_configs.py` as `AblationContract` objects. The currently supported canonical modes are:

- `full`
- `w_o_roi`
- `w_o_incongruity`
- `w_o_retrieval`
- `w_o_context_generation`
- `w_o_relevance_scorer`
- `w_o_support_verifier`
- `w_o_temporal_cultural_validator`
- `w_o_task_aware_gate`
- `w_o_structured_auxiliary`
- `label_only_no_evidence`

Alias:

- `w_o_verifier` resolves to `w_o_support_verifier`.

`label_only_no_evidence` is marked unsupported for default causal suites because the current implementation can hide final evidence/rationale fields but cannot guarantee that Stage C/D evidence did not influence labels without changing architecture.

## Audit Behavior

Full `ours_full` runs remain strict:

- six active logits losses are expected
- target presence and tactic multimodal relation losses must be differentiable
- Stage E provenance and evidence attribution fields are checked

Ablations are audited against their manifest contract:

- `w_o_retrieval`: Stage D verified knowledge count must be zero
- `w_o_context_generation`: Stage B generated candidate count must be zero
- `w_o_task_aware_gate`: Stage D must report shared-gate ablation mode
- `w_o_structured_auxiliary`: exactly four active logits losses are expected, with `target_presence` and `tactic_multimodal_relation` intentionally disabled

Evaluation-time ablations do not require a training log, but their prediction artifacts and component-state manifest are still audited.

## Resume And Skip-Complete

Use:

```bash
python scripts/run.py suite --suite core_1seed --resume
```

or:

```bash
python scripts/run.py suite --suite core_1seed --skip-complete
```

A run is skipped only when its run manifest, metrics, and predictions exist and the run is considered complete. Full-framework and ablation runs are re-audited before being skipped.
