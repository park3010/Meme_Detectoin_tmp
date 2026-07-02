# Experiment Protocol Lock

This repository uses `scripts/run.py suite` as the canonical entry point for reproducible paper experiment batches. The suite runner resolves a preset from `configs/config.yaml`, pins one split file per dataset/seed, executes the existing training or evaluation functions, writes run manifests, and optionally audits each full-framework or ablation run.

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

