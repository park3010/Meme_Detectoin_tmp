# Human Evaluation Protocol

The software exports blinded forms but does not generate ratings.

```bash
python scripts/run.py research human-export --suite harmeme_to_fhm_1seed --experiment ours_full --seed 42 --limit 100
```

Outputs under `human_eval/export/` include evidence, rationale, and verifier templates, a separate blinding key, and an export manifest. Rating schemas are:

- `human_eval/schemas/evidence.json`
- `human_eval/schemas/rationale.json`
- `human_eval/schemas/verifier.json`

Validate a completed sheet:

```bash
python scripts/run.py research human-validate \
  --input ratings.csv \
  --schema human_eval/schemas/rationale.json
```

Compute pairwise agreement only after validation:

```bash
python scripts/run.py research human-agreement \
  --input ratings.csv \
  --rating-column faithfulness usefulness specificity label_consistency
```

The validator checks required columns, duplicate item/annotator pairs, annotator IDs, and one-to-five ranges. Agreement reports exact agreement and quadratic weighted kappa; no value is emitted when paired data is insufficient.
