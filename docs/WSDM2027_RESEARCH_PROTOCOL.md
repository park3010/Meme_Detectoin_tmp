# WSDM 2027 Research Protocol

## Locked Domain Roles

The canonical paper protocol is `harmeme_to_fhm_v1`.

| Repository dataset | Paper name | Role | Structured provenance |
|---|---|---|---|
| `harm_c` | HarMeme COVID-19 | source train/validation | normalized clean labels when eligible |
| `harm_p` | HarMeme politics | source train/validation | normalized clean labels when eligible |
| `facebook` | FHM | held-out target test only | agent-silver evaluation labels |
| `memotion` | Memotion | disabled for paper runs | not applicable |

All 7,013 HarMeme rows with original binary labels are eligible for harmfulness training/evaluation. Structured losses are mask-aware and use only clean-eligible normalized fields. All 9,000 FHM rows are eligible for harmfulness testing; structured FHM metrics report valid-N and coverage.

The source split is fixed at seed 42 and stratified by original HarMeme dataset and harmfulness. Model seeds are `42, 52, 123, 777, 2026`; they never regenerate the split. Thresholds, early stopping, prompt choices, and model selection use HarMeme validation only.

## Immutable Inputs

- `result/splits/harmeme/source_split_seed_42.json`
- `result/splits/fhm/heldout_test_manifest.json`
- Sidecar SHA-256 files adjacent to both manifests
- `configs/experiment_registry.yaml`
- `configs/label_vocab.yaml`

Existing manifests are not replaced unless `--force-regenerate-split` is explicitly supplied. Every canonical run records split, config, code-tree, normalized-label, vocabulary, and asset provenance.

## Leakage Rules

FHM is forbidden from training, validation, early stopping, threshold selection, few-shot demonstrations, prompt/configuration development, and retrieval databases. Strict preflight currently blocks the repository because `dataset/source/wiki_common/wiki_manifest.json` declares Facebook provenance. Rebuild that corpus from HarMeme-train-safe/general sources and update its manifest before executing paper suites.

Run:

```bash
python scripts/run.py research preflight --strict
```

The machine-readable and Markdown reports are written under `result/research_planning/`.
