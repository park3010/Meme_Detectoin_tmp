# Research CLI

The `research` command group is additive; legacy CLI commands remain available.

```bash
# Resolve registry/suite only
python scripts/run.py research plan --suite harmeme_to_fhm_smoke

# Create/validate immutable manifests and run strict leakage checks
python scripts/run.py research preflight --strict

# Explicit dry plan; no training
python scripts/run.py research run --suite harmeme_to_fhm_smoke --dry-run

# A selected suite executes only after passing strict preflight
python scripts/run.py research run --suite harmeme_to_fhm_smoke --device cuda

# Select an explicit registered subset from a suite
python scripts/run.py research run \
  --suite harmeme_to_fhm_1seed \
  --experiment ours_full ablation_w_o_retrieval \
  --device cuda

# Resume skips audited-complete runs and retries partial/failed run directories
python scripts/run.py research run --suite harmeme_to_fhm_1seed --resume --device cuda

python scripts/run.py research status
python scripts/run.py research audit --strict
python scripts/run.py research aggregate
python scripts/run.py research dashboard
python scripts/run.py research paper-export
python scripts/run.py research paper-check --mode draft
```

`--limit` applies independently to source train, source validation, and FHM test selections for smoke work. It must not be used for final paper runs. Existing run directories are not overwritten unless `--force` is explicit; `--resume` is the deliberate exception for retrying incomplete runs. `--experiment` accepts only IDs that belong to the selected suite.

Canonical runs live at:

```text
result/research_runs/<suite>/<experiment_id>/seed_<seed>/
```

Each completed run includes manifests, resolved configuration, environment, logs, validation/test predictions, metrics, thresholds, complexity, runtime, and JSON/Markdown audits.
