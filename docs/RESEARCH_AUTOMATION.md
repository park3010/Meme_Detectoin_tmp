# Research Automation

The canonical path is `scripts/run.py` to `scripts/commands/research.py`, then `experiments/research_orchestration.py`, the registry, and an experiment adapter. Built-in adapters call the existing trainers; blocked adapters never execute.

Planning and dry runs are side-effect-light. Actual suite execution requires an explicit suite and first runs strict preflight. Resume skips only audited-complete runs. Canonical artifacts and outputs are documented in [RESEARCH_CLI.md](RESEARCH_CLI.md) and [RESEARCH_IMPLEMENTATION_MAP.md](RESEARCH_IMPLEMENTATION_MAP.md).

Retrieval has three dedicated protocol commands:

```bash
conda run -n meme_cikm python scripts/run.py research retrieval-build --offline
conda run -n meme_cikm python scripts/run.py research retrieval-audit --strict
conda run -n meme_cikm python scripts/run.py research retrieval-status
```

The build is deterministic, offline, and non-overwriting by default. A matching
existing output is reported as already valid; a differing output requires
`--force` and is replaced atomically. `--limit` is reserved for tests/smoke and
produces a non-paper-eligible manifest. Full and retrieval-enabled paper runs
fail before Stage B if the active profile or any declared hash is invalid.
