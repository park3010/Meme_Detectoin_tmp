# Repository Consolidation Refactor Plan

This refactor reduces implementation fragmentation while preserving model behavior, output schemas, provenance contracts, dataset semantics, and result paths.

## Migration Map

| Area | Previous source layout | Consolidated source |
| --- | --- | --- |
| Stage A | `module/stage_a/*` | `module/internal_evidence_extractor.py` |
| Stage B | `module/stage_b/*` | `module/external_knowledge_acquisition.py` |
| Stage C | `module/stage_c/*` | `module/knowledge_filter_verifier.py` |
| Stage D | `module/stage_d/*` | `module/evidence_fusion_reasoning.py` |
| Stage E | `module/stage_e/*` | `module/structured_interpretation_head.py` |
| Backbones | `module/backbones/*` | `module/backbone/{vision,text,retrieval,generation}.py` |
| Baselines | `module/baselines/*` | `module/baseline.py` |
| Losses | `module/losses/*` | `module/losses.py` |
| Pipeline | `module/pipeline/*` | `module/runner.py` |
| Normalized labels | `dataset/normalized_labels.py`, `dataset/label_adapter.py` | `dataset/labels.py` |
| Training | `experiments/train_ours.py`, `experiments/train_baseline.py` | `experiments/train.py` |
| Evaluation | `experiments/metrics.py`, `experiments/structured_eval.py`, `experiments/evaluate_predictions.py` | `experiments/evaluation.py` |

## Active Entry Points

- Full framework training: `python scripts/run.py train ...`
- Baseline training: `python scripts/run.py baseline ...`
- Stage-only execution: `python scripts/run.py stage ...`
- Structured evaluation: `python scripts/run.py evaluate ...`
- Ablation/fusion: `python scripts/run.py ablation ...`
- Artifact audit: `python scripts/run.py audit ...`

## Cleanup Note

The consolidated modules are now active. Removing obsolete duplicate directories/scripts requires explicit deletion approval in this sandbox because the broad cleanup command was blocked by the safety layer.
