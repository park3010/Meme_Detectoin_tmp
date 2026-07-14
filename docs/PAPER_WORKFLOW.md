# Paper and Dashboard Workflow

The WSDM scaffold is under `latex/`. `main.tex` uses anonymous ACM `sigconf,review` mode and imports section files plus generated tables. Hand-written content and generated artifacts are separated.

```bash
python scripts/run.py research aggregate
python scripts/run.py research dashboard
python scripts/run.py research paper-export
make -C latex draft
python scripts/run.py research paper-check --mode draft
```

The dashboard is self-contained at `result/dashboard/index.html`. Paper exports write only below `latex/generated/`. Numeric tables are generated from canonical completed results. Before experiments, safe `Not run`/`--` rows are used instead of fabricated values.

Final readiness is stricter:

```bash
python scripts/run.py research paper-check --mode final
```

Final mode fails when draft markers, blocked rows, or missing required files remain. The bibliography is preserved in `latex/reference.bib`; citation-backed prose and final conference metadata require author review.
