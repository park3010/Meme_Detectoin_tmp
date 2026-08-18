# Experiment Master Plan

The six registered paper families are E1 baseline comparison, E2 train-time core ablation, E3 train-time knowledge comparison, E4 silver structured interpretation, E5 evidence/rationale/verifier evaluation, and E6 error analysis.

Execution order is: build/audit the frozen HarMeme-train retrieval resource,
strict protocol preflight, framework smoke, one-seed framework pilot,
aggregation/audit, then five seeds. Knowledge modes remain diagnostic until
train-time variants exist. External and human/API methods remain explicit
blockers.

The current framework server uses the dedicated registry suites
`ours_framework_smoke`, `ours_framework_1seed`, and `ours_framework_5seed`.
They contain only `ours_full` and the four paper core ablations: no built-in
baseline, external model, or diagnostic knowledge condition. The comparison
server owns the four built-in baselines, external SOTA methods, and future
LMM/VLM/agent comparisons. The legacy `harmeme_to_fhm_*` combined suites remain
registered for historical and combined-server workflows, but are not the
default execution path on the framework-only server.

The smoke suite reuses the canonical audit-valid limit of 100. Run it under an
isolated output root and pass `--epochs 1` for short smoke training; pilot and
five-seed suites have no suite-level epoch or limit override.

The canonical resource is `dataset/retrieval/harmeme_train_v1/`. Do not start a
retrieval-enabled suite unless `research retrieval-audit --strict` and
`research preflight --strict` both pass. The `w_o_retrieval` contract may run
without this resource because it does not instantiate a corpus-backed Stage B.

See [WSDM2027_RESEARCH_PROTOCOL.md](WSDM2027_RESEARCH_PROTOCOL.md), [RESEARCH_REGISTRY.md](RESEARCH_REGISTRY.md), and [RESEARCH_CLI.md](RESEARCH_CLI.md).
