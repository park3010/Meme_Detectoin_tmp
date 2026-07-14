# External Baseline Policy

External methods are represented by explicit feasibility records in `configs/external_models.yaml` and `result/research_planning/external_baseline_feasibility.json`.

The repository does not automatically clone repositories, download checkpoints, invoke proprietary APIs, accept licenses, or install conflicting environments. Those actions require separate approval and isolated environments. A method is not marked ready unless its official implementation, exact checkpoint, license, dependencies, adapter, and HarMeme-to-FHM protocol mapping are verified.

Blocked entries are shown as blocked or not run in dashboards and LaTeX. They are never filled with zero metrics. Prompting variants cannot inspect FHM labels or examples during prompt development. API variants remain blocked without credentials and an approved cost/privacy plan.
