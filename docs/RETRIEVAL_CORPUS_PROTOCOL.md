# Retrieval Corpus Protocol

## Scope

The canonical paper profile is `harmeme_train_v1`, with role
`harmeme_train_conditioned`. Corpus and index construction may use only exact
sample keys in the immutable HarMeme `train` partition from:

```text
result/splits/harmeme/source_split_seed_42.json
```

HarMeme validation, Facebook/FHM, Memotion, and unverifiable sample origins are
forbidden. The current split-manifest file SHA-256 is:

```text
f02364f41169fdeac4ec1cffe68192b9d522e7fe5bca5ba2848ad6dfe777200c
```

## Construction Versus Querying

Corpus construction decides which external documents exist in the frozen
index. It is dataset-conditioned and therefore accepts only auditable source
train queries. Test-time querying only ranks those already frozen documents.
FHM may query at final inference, but it cannot add documents, alter index
weights, choose hyperparameters, or expose gold labels to retrieval code.

FHM query-result logs are run-specific:

```text
result/.../retrieval_queries/fhm/queries.jsonl
```

They are not corpus provenance and are never written under `dataset/retrieval/`.

## Offline Replay

The original aggregate Wikipedia corpus was selected using mixed dataset
provenance. It is not filtered or copied. The canonical builder instead:

1. Reads the immutable source manifest and selects exact `train` sample keys.
2. Reads only the Harm-C and Harm-P per-query caches.
3. Ignores cached labels and legacy split labels.
4. Retains a document only when an eligible train query independently returned
   its real `kid`.
5. Joins that `kid` to the local Wikipedia sentence snapshot for title/source
   metadata.
6. Deduplicates documents while merging only legitimate train-query origins.
7. Rebuilds sparse token statistics and 256-dimensional deterministic hashed
   dense vectors from the new corpus.

All 5,611 source-train samples have auditable cache rows. The current replay
contains 4,791 unique documents from 28,055 eligible retrieval occurrences.
No network is used.

## Layout

```text
dataset/retrieval/harmeme_train_v1/
├── corpus/corpus_texts.jsonl
├── index/sparse/
├── index/dense/
├── cache/source_queries/queries.jsonl
├── cache/source_documents/documents.jsonl
├── retrieval_manifest.json
├── index_manifest.json
└── checksums.sha256
```

Each corpus row records the external document identifier/text/source plus
origin sample keys, original datasets, domains, query IDs/types, and the fixed
`train` partition. Query IDs are deterministic hashes of the real dataset,
sample ID, and exact cached query. They do not replace or fabricate sample IDs.

## Manifests And Hashes

`retrieval_manifest.json` uses `retrieval_corpus_manifest_v2`. It records the
source split path/hash, partition, included/excluded datasets, query coverage,
document count, corpus hash, build/config/code provenance, network policy, and
forbidden-origin counts.

`index_manifest.json` binds both indexes to the exact corpus and retrieval
manifest. It records sparse/dense backends, embedding specification, indexed
document count, and directory hashes. Runtime rejects any mismatch in corpus,
manifest, index, or count.

Current canonical content hashes are:

```text
corpus  7f11e3af73118795500e488a5fc090d39187defd422d7965d0de5cc61ec3afb7
sparse  3e81560db8aed76012a652628d066a867150172e67aefa8ed1ac33e7f11a3ecd
dense   a597632b996a2b1f9d26c4bd86f853f529b024f04e93d6e0d3d05adad5559f67
```

Build timestamps follow reproducible-build convention: `SOURCE_DATE_EPOCH` is
used when supplied and Unix epoch otherwise. Corpus/index content hashes remain
deterministic for identical inputs.

## Commands

```bash
conda run -n meme_cikm python scripts/run.py research retrieval-build \
  --registry configs/experiment_registry.yaml \
  --profile harmeme_train_v1 \
  --source-partition train \
  --output-root dataset/retrieval/harmeme_train_v1 \
  --offline

conda run -n meme_cikm python scripts/run.py research retrieval-audit \
  --registry configs/experiment_registry.yaml \
  --profile harmeme_train_v1 \
  --strict

conda run -n meme_cikm python scripts/run.py research preflight \
  --registry configs/experiment_registry.yaml \
  --strict
```

A matching existing build is left untouched. A differing directory is refused
unless `--force` is explicit; replacement uses a sibling temporary directory
and atomic rename. `--limit` creates a test-only, non-paper build.

## Legacy Artifact

`dataset/source/wiki_common/` is preserved unchanged for historical/debug
reproduction. It is configured as `legacy_wiki_common` with role
`legacy_non_paper`, `enabled: false`, and `paper_eligible: false`. Its manifest
contains Facebook and Memotion provenance and is not bound to the immutable
source split. It must never pass strict audit or become an automatic fallback.

## Static External Corpus Mode

A future resource may use role `static_external_dataset_independent` only if
construction evidence proves that no experimental dataset query, label, or
sample selected its contents. It still requires corpus/index manifests and
hashes. The current legacy Wikipedia resource does not satisfy this condition
and must not be relabeled.

## Troubleshooting

- `retrieval_source_split_hash_mismatch`: restore the immutable split artifact;
  do not regenerate or edit its hash in the retrieval manifest.
- `retrieval_corpus_hash_mismatch`: rebuild from local query caches; do not edit
  the declared hash by hand.
- `retrieval_index_hash_mismatch`: rebuild both indexes from the canonical
  corpus; never copy the legacy embedding/index.
- `retrieval_validation_leakage`, `fhm_retrieval_leakage`, or
  `disabled_dataset_retrieval_leakage`: inspect row/query origins and stop.
- `blocked_retrieval_corpus_rebuild`: required row-level/query-level provenance
  is missing. Do not filter an aggregate mixed corpus as a workaround.
