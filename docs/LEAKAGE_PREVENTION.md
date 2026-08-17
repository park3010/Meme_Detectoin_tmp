# Leakage Prevention

FHM is prohibited from training, validation, early stopping, threshold tuning, prompt/few-shot development, configuration selection, and retrieval indexes. HarMeme validation examples are query-only and are not labeled neighbors in a paper retrieval index. Memotion is excluded from all paper suites.

The strict audit scans source/FHM manifest overlap and retrieval-corpus provenance. Dataset-derived indexes must declare `source_partition: harmeme_train` (or `train`) and bind `source_split_content_sha256` to the immutable source manifest. It currently fails because the configured wiki corpus manifest includes Facebook and disabled Memotion, does not declare a train-only partition, and is not bound to the source split. Do not weaken the audit; rebuild the corpus with leakage-safe provenance before execution.
