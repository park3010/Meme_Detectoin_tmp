# WSDM 2027 Research Protocol

## Locked Domain Roles

The canonical paper protocol is `harmeme_to_fhm_v2`.

| Repository dataset | Paper name | Role | Structured provenance |
|---|---|---|---|
| `harm_c` | HarMeme COVID-19 | source train/validation | normalized clean labels when eligible |
| `harm_p` | HarMeme politics | source train/validation | normalized clean labels when eligible |
| `facebook` | FHM | held-out target test only | agent-silver evaluation labels |
| `memotion` | Memotion | disabled for paper runs | not applicable |

HarMeme contains 7,013 raw source records. After a source-only integrity audit, we exclude 72 redundant, raw-label-conflicting, or structured-label-conflicting records and construct a group-aware source split over 6,941 retained samples. Structured losses are mask-aware and use only clean-eligible normalized fields. Raw-source conflict groups are excluded in full because no supervised annotation can be selected without adjudicating against the source. Decoded-image groups are split atomically; source-confirmed shared-image/different-text records remain distinct multimodal samples.

The v2 source split is fixed at seed 42, group-aware by decoded image, and approximately stratified by original HarMeme dataset and harmfulness. It contains 5,552 training and 1,389 validation records. Model seeds are `42, 52, 123, 777, 2026`; they never regenerate the split. Thresholds, early stopping, prompt choices, and model selection use HarMeme validation only. Split-v1 results are historical engineering artifacts and are not paper eligible.

## Immutable Inputs

- `result/splits/harmeme/source_split_seed_42.json`
- `result/splits/fhm/heldout_test_manifest.json`
- Sidecar SHA-256 files adjacent to both manifests
- `configs/experiment_registry.yaml`
- `configs/label_vocab.yaml`

Existing manifests are not replaced unless `--force-regenerate-split` is explicitly supplied. Every canonical run records split, config, code-tree, normalized-label, vocabulary, and asset provenance.

## Leakage Rules

FHM is forbidden from training, validation, early stopping, threshold selection, few-shot demonstrations, prompt/configuration development, and retrieval databases. Strict preflight currently blocks the repository because `dataset/source/wiki_common/wiki_manifest.json` declares Facebook and Memotion provenance and lacks immutable HarMeme-train-only split binding. Rebuild that corpus from HarMeme-train-safe/general sources and update its manifest before executing paper suites.

Run:

```bash
python scripts/run.py research preflight --strict
```

The machine-readable and Markdown reports are written under `result/research_planning/`.
