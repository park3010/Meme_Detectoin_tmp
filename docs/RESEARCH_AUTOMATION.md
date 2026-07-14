# Research Automation

The canonical path is `scripts/run.py` to `scripts/commands/research.py`, then `experiments/research_orchestration.py`, the registry, and an experiment adapter. Built-in adapters call the existing trainers; blocked adapters never execute.

Planning and dry runs are side-effect-light. Actual suite execution requires an explicit suite and first runs strict preflight. Resume skips only audited-complete runs. Canonical artifacts and outputs are documented in [RESEARCH_CLI.md](RESEARCH_CLI.md) and [RESEARCH_IMPLEMENTATION_MAP.md](RESEARCH_IMPLEMENTATION_MAP.md).
