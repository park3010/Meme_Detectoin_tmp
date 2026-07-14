# Experiment Master Plan

The six registered paper families are E1 baseline comparison, E2 train-time core ablation, E3 train-time knowledge comparison, E4 silver structured interpretation, E5 evidence/rationale/verifier evaluation, and E6 error analysis.

Execution order is: strict protocol preflight, built-in smoke, one-seed built-ins, one-seed core ablations, aggregation/audit, then five seeds. Knowledge modes remain diagnostic until train-time variants exist. External and human/API methods remain explicit blockers.

See [WSDM2027_RESEARCH_PROTOCOL.md](WSDM2027_RESEARCH_PROTOCOL.md), [RESEARCH_REGISTRY.md](RESEARCH_REGISTRY.md), and [RESEARCH_CLI.md](RESEARCH_CLI.md).
