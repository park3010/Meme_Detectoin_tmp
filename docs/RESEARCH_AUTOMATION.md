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

## Server Roles And Framework Suites

Use these registry suites on the current framework server:

- `ours_framework_smoke`: `ours_full` plus four core train-time ablations,
  seed 42, and the canonical audit-valid limit of 100.
- `ours_framework_1seed`: the same five conditions at seed 42, without a smoke
  limit or epoch override.
- `ours_framework_5seed`: the same five conditions at seeds 42, 52, 123, 777,
  and 2026 (25 runs), without a smoke override.

The framework server is responsible for Ours Full, core ablations, future
paper-valid knowledge experiments, and structured/evidence/error analysis. A
separate comparison server is responsible for the four built-in baselines,
external SOTA methods, and LMM/VLM/agent models. Existing combined
`harmeme_to_fhm_*` suites remain available for historical or coordinated runs,
but they are not the framework server's default path.

Planning is safe and execution is explicit. Use a dedicated output root for a
smoke run:

```bash
conda run -n meme_cikm python scripts/run.py research run \
  --suite ours_framework_smoke \
  --epochs 1 \
  --device cuda \
  --output-root result/ours_smoke
```

Omit `--epochs 1` only when intentionally using the framework adapter's normal
training default. Strict preflight and formal tactic decoding are unchanged.
