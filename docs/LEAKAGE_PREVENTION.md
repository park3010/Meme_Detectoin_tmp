# Leakage Prevention

FHM is prohibited from training, validation, early stopping, threshold tuning, prompt/few-shot development, configuration selection, and retrieval indexes. HarMeme validation examples are query-only and are not labeled neighbors in a paper retrieval index. Memotion is excluded from all paper suites.

The strict audit scans source/FHM manifest overlap and retrieval-corpus provenance. It currently fails because the configured wiki corpus manifest includes Facebook. Do not weaken the audit; rebuild the retrieval corpus with leakage-safe provenance before execution.
