# External Baseline Feasibility

Authoritative identification references live in `configs/external_models.yaml`. The strict preflight exports one explicit feasibility row per registered external condition to `result/research_planning/external_baseline_feasibility.json`. No repository, checkpoint, dependency, environment, paid API, or model output is fetched or created by this automation.

## Current state

All external conditions remain blocked. Verified repository URLs identify projects only; they do not establish license compatibility, a reproducible commit, an exact checkpoint, environment compatibility, or HarMeme-to-FHM protocol validity. Missing information is represented as `unknown_*`, never guessed.

The machine-readable report records the model and experiment IDs, paper and repository references, repository-verification state, license and commit state, Python/PyTorch/CUDA requirements, checkpoints, datasets, preprocessing, GPU/storage/runtime estimates, protocol compatibility, adapter complexity, architecture-preservation risks, isolated environment, blockers, and next approval action.

## Approval gate

Before an adapter can become ready, a reviewer must approve isolated inspection of the official implementation, verify its license, pin the exact commit and checkpoint, preserve the original architecture/objective, document preprocessing, create an environment outside this source snapshot, and validate that only HarMeme train/validation informs development while FHM remains test-only. GPT-4o additionally requires an explicit cost/API approval and fixed model identifier; Qwen2.5-VL and LLaVA require exact checkpoint revisions.

Future approved repositories belong outside the snapshot at `/home/sujin/psj2003/meme_detection_external/<model_id>`. See [EXTERNAL_BASELINE_POLICY.md](EXTERNAL_BASELINE_POLICY.md).
