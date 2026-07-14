# Dataset Protocol: HarMeme to FHM

`harm_c` and `harm_p` form HarMeme source train/validation while preserving COVID-19 and politics domain metadata. `facebook` is FHM and is test-only. `memotion` is disabled.

The fixed seed-42 split is approximately 80/20, jointly stratified by original dataset and harmfulness. Model seeds never alter it. Harmfulness uses original binary labels; structured fields use clean-eligible masks and FHM agent-silver evaluation provenance.

See [WSDM2027_RESEARCH_PROTOCOL.md](WSDM2027_RESEARCH_PROTOCOL.md) for hashes, paths, and commands.
