# Research Implementation Map

## Protocol and Registry

- `dataset/meme_dataset.py`: dataset-family/domain-role metadata.
- `experiments/research_protocol.py`: immutable split manifests and FHM leakage audit.
- `configs/experiment_registry.yaml`: experiments and suites.
- `configs/external_models.yaml`: external feasibility catalog.
- `experiments/registry.py`: normalization and validation.

## Execution

- `scripts/run.py research ...`: unified command entry point.
- `scripts/commands/research.py`: argument parsing.
- `experiments/research_orchestration.py`: plan, preflight, run, resume, status, audit.
- `experiments/adapters/`: lifecycle contract, built-in runners, blocked external adapter.
- `experiments/train.py`: existing trainers with additive source/held-out manifest support.

## Reporting

- `experiments/research_results.py`: canonical long-form and seed aggregation.
- `experiments/research_dashboard.py`: static offline dashboard.
- `experiments/research_error_analysis.py`: auditable FHM case packages.
- `experiments/research_human_eval.py`: blinded forms, validation, agreement.
- `experiments/paper_export.py`: generated LaTeX inputs and paper checks.

The Stage A--E architecture remains unchanged. The only added model is the registered supervised OpenCLIP baseline in `module/baseline.py`, supported by an additive CLIP text-encoding method in `module/backbone/vision.py`.
