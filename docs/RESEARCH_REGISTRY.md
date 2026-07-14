# Research Experiment Registry

`configs/experiment_registry.yaml` is the canonical inventory of built-in, ablation, knowledge, external, and prompting experiments. `configs/external_models.yaml` records external-method feasibility without downloading or claiming an implementation.

Each normalized experiment has an ID, family, group, adapter, status, domain roles, tasks, seeds, split manifest, dependencies, implementation policy, and paper targets. Valid statuses include `ready`, `completed`, `blocked_dependency`, `blocked_checkpoint`, `blocked_api_credentials`, and `unsupported_current_protocol`.

Ready built-ins are:

- `text_only_deberta`
- `image_only_openclip`
- `image_text_concat`
- `openclip_classifier`
- `ours_full`
- Four train-time core ablations

The registered knowledge comparison is intentionally `unsupported_current_protocol`: the pre-existing knowledge runner applies evaluation-time transformations and is not a valid substitute for separately trained protocol variants.

Inspect without running:

```bash
python scripts/run.py research plan
python scripts/run.py research plan --suite harmeme_to_fhm_1seed
```

Blocked external entries are planning/reporting records. They never execute through the built-in adapter.
