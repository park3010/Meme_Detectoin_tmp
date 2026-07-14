# Research Reproducibility Checklist

- [ ] `python -m pytest tests -q` passes.
- [ ] `python scripts/run.py research preflight --strict` passes.
- [ ] Retrieval manifests prove that FHM/Facebook examples and annotations are absent.
- [ ] Source and FHM manifest SHA-256 sidecars are archived.
- [ ] All paper runs use source split seed 42 and the registered model seeds.
- [ ] HarMeme validation is the only model/threshold-selection source.
- [ ] FHM is read only during final held-out prediction/evaluation.
- [ ] Every run has all canonical artifacts and a passing pipeline audit.
- [ ] Structured metrics report valid-N, total-N, coverage, unknown, ambiguous, and masked counts.
- [ ] Formal tactic metrics use logits-only predictions and a frozen validation threshold.
- [ ] External methods are either verified adapters or explicit blocked records.
- [ ] Human ratings are validated and agreement is reported; unrated templates are not treated as results.
- [ ] `python scripts/run.py research paper-check --mode final` passes.
- [ ] `make -C latex draft` produces the anonymous review PDF.
