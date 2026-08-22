# Dataset Protocol: HarMeme to FHM

`harm_c` and `harm_p` form HarMeme source train/validation while preserving COVID-19 and politics domain metadata. The v2 effective supervised pool contains 6,941 of 7,013 raw records after 72 duplicate/conflict exclusions. Decoded-image groups are indivisible across partitions, while shared-image/different-text source records remain distinct. `facebook` is FHM and is test-only. `memotion` is disabled.

The fixed seed-42 split is approximately 80/20, jointly stratified by original dataset and harmfulness. Model seeds never alter it. Harmfulness uses original binary labels; structured fields use clean-eligible masks and FHM agent-silver evaluation provenance.

See [WSDM2027_RESEARCH_PROTOCOL.md](WSDM2027_RESEARCH_PROTOCOL.md) for hashes, paths, and commands.

## Retrieval boundary

The paper retrieval corpus is bound to the same immutable source manifest and
is reconstructed only from its 5,552 v2 `train` rows. HarMeme validation, FHM, and
Memotion cannot create corpus documents or index entries. FHM may issue
unlabeled queries to the already frozen index during final inference; those
query-result records are written beneath the run output and never under the
canonical corpus root.

See [RETRIEVAL_CORPUS_PROTOCOL.md](RETRIEVAL_CORPUS_PROTOCOL.md) for the
construction, manifest, hash, and audit contract.
