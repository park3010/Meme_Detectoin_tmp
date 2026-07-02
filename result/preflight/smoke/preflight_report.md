# Experiment 0 Preflight Report

## Profile and command
- Profile: `smoke`
- Strict: `False`
- Config: `configs/config.yaml`
- Config SHA256: `f7bd293ca939a0c6cc8d76440d3f4b2ec4cb67de1b1fe28df6c2b590e53f7556`
- Created: `2026-07-02T16:07:42.692805+00:00`

## Overall decision
**PASS_WITH_WARNINGS**

## Backbone readiness
- Vision main-ready: `True`
- Text main-ready: `True`
- Vision state: `{'requested_backend': 'clip', 'resolved_backend': 'fallback', 'model_name': 'ViT-B-32', 'prefer_pretrained': True, 'pretrained_requested': True, 'pretrained_tag': None, 'checkpoint_path': None, 'checkpoint_exists': False, 'weights_loaded': False, 'weights_source': None, 'local_files_only': True, 'allow_download': False, 'fallback_used': True, 'random_initialization_used': False, 'load_error': 'pretrained vision requested, but no local checkpoint_path was configured'}`
- Text state: `{'requested_backend': 'transformers', 'resolved_backend': 'hashing', 'model_name': 'microsoft/deberta-v3-base', 'checkpoint_path': None, 'checkpoint_exists': False, 'weights_loaded': False, 'weights_source': None, 'local_files_only': True, 'allow_download': False, 'fallback_used': True, 'load_error': "OSError: We couldn't connect to 'https://huggingface.co' to load this file, couldn't find it in the cached files and it looks like microsoft/deberta-v3-base is not the path to a directory containing a file named config.json.\nCheckout your internet connection or see how to run the library in offline mode at '"}`

## Dataset / label eligibility
- Datasets: `['harm_c', 'harm_p', 'facebook', 'memotion']`
- Detailed CSV: `result/preflight/smoke/dataset_metric_eligibility.csv`

## Split integrity
- Split report: `result/preflight/smoke/split_integrity_report.json`

## Retrieval corpus readiness
- Usable corpora: `1`
- Provenance note: `fallback candidate != retrieved external knowledge; generated hypothesis != retrieved external knowledge`

## Metric contract
- Implementation status: `partially_ready`
- Missing capabilities: `['current structured evaluator consumes rendered tactic.rhetorical labels', 'validation-selected sigmoid threshold path for tactic logits is not yet exposed']`

## Normalized annotation snapshot
- Snapshot artifact: `result/preflight/smoke/normalized_annotation_snapshot.json`

## Warnings
- `vision_fallback_backbone`: vision backbone is using fallback features. `{'requested_backend': 'clip', 'resolved_backend': 'fallback', 'model_name': 'ViT-B-32', 'prefer_pretrained': True, 'pretrained_requested': True, 'pretrained_tag': None, 'checkpoint_path': None, 'checkpoint_exists': False, 'weights_loaded': False, 'weights_source': None, 'local_files_only': True, 'allow_download': False, 'fallback_used': True, 'random_initialization_used': False, 'load_error': 'pretrained vision requested, but no local checkpoint_path was configured'}`
- `text_fallback_backbone`: text backbone is using fallback features. `{'requested_backend': 'transformers', 'resolved_backend': 'hashing', 'model_name': 'microsoft/deberta-v3-base', 'checkpoint_path': None, 'checkpoint_exists': False, 'weights_loaded': False, 'weights_source': None, 'local_files_only': True, 'allow_download': False, 'fallback_used': True, 'load_error': "OSError: We couldn't connect to 'https://huggingface.co' to load this file, couldn't find it in the cached files and it looks like microsoft/deberta-v3-base is not the path to a directory containing a file named config.json.\nCheckout your internet connection or see how to run the library in offline mode at '"}`
- `metric_low_support`: Dataset/field passes hard eligibility but is below advisory support. `{'dataset': 'harm_c', 'seed': 42, 'field': 'intent_primary'}`
- `metric_low_support`: Dataset/field passes hard eligibility but is below advisory support. `{'dataset': 'harm_c', 'seed': 42, 'field': 'tactic_rhetorical'}`
- `metric_low_support`: Dataset/field passes hard eligibility but is below advisory support. `{'dataset': 'harm_c', 'seed': 42, 'field': 'tactic_multimodal_relation'}`
- `metric_low_support`: Dataset/field passes hard eligibility but is below advisory support. `{'dataset': 'harm_p', 'seed': 42, 'field': 'intent_primary'}`
- `metric_low_support`: Dataset/field passes hard eligibility but is below advisory support. `{'dataset': 'memotion', 'seed': 42, 'field': 'tactic_multimodal_relation'}`
- `retrieval_fallback_enabled`: Fallback candidates are enabled; provenance must not treat them as retrieved external knowledge. `{}`

## Blocking errors
- None

## Main-experiment acceptance decision
Main experiment results are valid only after `main_experiment` strict preflight passes.

## Required next actions
- None
