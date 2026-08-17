# Experiment Master Plan

The six registered paper families are E1 baseline comparison, E2 train-time core ablation, E3 train-time knowledge comparison, E4 silver structured interpretation, E5 evidence/rationale/verifier evaluation, and E6 error analysis.

Execution order is: build/audit the frozen HarMeme-train retrieval resource,
strict protocol preflight, built-in smoke, one-seed built-ins, one-seed core
ablations, aggregation/audit, then five seeds. Knowledge modes remain
diagnostic until train-time variants exist. External and human/API methods
remain explicit blockers.

The canonical resource is `dataset/retrieval/harmeme_train_v1/`. Do not start a
retrieval-enabled suite unless `research retrieval-audit --strict` and
`research preflight --strict` both pass. The `w_o_retrieval` contract may run
without this resource because it does not instantiate a corpus-backed Stage B.

See [WSDM2027_RESEARCH_PROTOCOL.md](WSDM2027_RESEARCH_PROTOCOL.md), [RESEARCH_REGISTRY.md](RESEARCH_REGISTRY.md), and [RESEARCH_CLI.md](RESEARCH_CLI.md).
