# Leakage Prevention

FHM is prohibited from training, validation, early stopping, threshold tuning, prompt/few-shot development, configuration selection, and retrieval indexes. HarMeme validation examples are query-only and are not labeled neighbors in a paper retrieval index. Memotion is excluded from all paper suites.

The strict audit scans source/FHM manifest overlap, row-level retrieval origins,
the immutable split hash, corpus hash, corpus-manifest hash, sparse/dense index
hashes, and indexed document count. The active paper profile is
`harmeme_train_v1`; it declares `source_partition: train` and binds the exact
source-manifest file SHA-256. Any validation, FHM, Memotion, or unknown origin
is a hard failure.

`dataset/source/wiki_common/` remains an unchanged historical artifact. Its
aggregate provenance includes Facebook and Memotion, so configuration marks it
`legacy_non_paper` and strict audit always rejects it. Runtime does not silently
fall back to that tree. A true `w_o_retrieval` run resolves no corpus at all.

Construction provenance and inference query provenance are separate. An FHM
test query may search the frozen HarMeme-train-only index, but it cannot mutate
the corpus/index and its labels are never passed to retrieval. Query logs live
under `result/.../retrieval_queries/fhm/`.
