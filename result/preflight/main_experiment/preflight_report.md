# Experiment 0 Preflight Report

## Profile and command
- Profile: `main_experiment`
- Strict: `True`
- Config: `configs/config.yaml`
- Config SHA256: `f7bd293ca939a0c6cc8d76440d3f4b2ec4cb67de1b1fe28df6c2b590e53f7556`
- Created: `2026-07-02T16:07:59.214248+00:00`

## Overall decision
**BLOCKED**

## Backbone readiness
- Vision main-ready: `False`
- Text main-ready: `False`
- Vision state: `{'requested_backend': 'clip', 'resolved_backend': 'fallback', 'model_name': 'ViT-B-32', 'prefer_pretrained': True, 'pretrained_requested': True, 'pretrained_tag': None, 'checkpoint_path': None, 'checkpoint_exists': False, 'weights_loaded': False, 'weights_source': None, 'local_files_only': True, 'allow_download': False, 'fallback_used': True, 'random_initialization_used': False, 'load_error': 'pretrained vision requested, but no local checkpoint_path was configured'}`
- Text state: `{'requested_backend': 'transformers', 'resolved_backend': 'hashing', 'model_name': 'microsoft/deberta-v3-base', 'checkpoint_path': None, 'checkpoint_exists': False, 'weights_loaded': False, 'weights_source': None, 'local_files_only': True, 'allow_download': False, 'fallback_used': True, 'load_error': "OSError: We couldn't connect to 'https://huggingface.co' to load this file, couldn't find it in the cached files and it looks like microsoft/deberta-v3-base is not the path to a directory containing a file named config.json.\nCheckout your internet connection or see how to run the library in offline mode at '"}`

## Dataset / label eligibility
- Datasets: `['harm_c', 'harm_p', 'facebook', 'memotion']`
- Detailed CSV: `result/preflight/main_experiment/dataset_metric_eligibility.csv`

## Split integrity
- Split report: `result/preflight/main_experiment/split_integrity_report.json`

## Retrieval corpus readiness
- Usable corpora: `1`
- Provenance note: `fallback candidate != retrieved external knowledge; generated hypothesis != retrieved external knowledge`

## Metric contract
- Implementation status: `partially_ready`
- Missing capabilities: `['current structured evaluator consumes rendered tactic.rhetorical labels', 'validation-selected sigmoid threshold path for tactic logits is not yet exposed']`

## Normalized annotation snapshot
- Snapshot artifact: `result/preflight/main_experiment/normalized_annotation_snapshot.json`

## Warnings
- `metric_low_support`: Dataset/field passes hard eligibility but is below advisory support. `{'dataset': 'harm_c', 'seed': 42, 'field': 'intent_primary'}`
- `metric_low_support`: Dataset/field passes hard eligibility but is below advisory support. `{'dataset': 'harm_c', 'seed': 42, 'field': 'tactic_rhetorical'}`
- `metric_low_support`: Dataset/field passes hard eligibility but is below advisory support. `{'dataset': 'harm_c', 'seed': 42, 'field': 'tactic_multimodal_relation'}`
- `metric_low_support`: Dataset/field passes hard eligibility but is below advisory support. `{'dataset': 'harm_p', 'seed': 42, 'field': 'intent_primary'}`
- `metric_low_support`: Dataset/field passes hard eligibility but is below advisory support. `{'dataset': 'memotion', 'seed': 42, 'field': 'tactic_multimodal_relation'}`
- `retrieval_fallback_enabled`: Fallback candidates are enabled; provenance must not treat them as retrieved external knowledge. `{}`

## Blocking errors
- `vision_fallback_backbone`: vision backbone is using fallback features. `{'requested_backend': 'clip', 'resolved_backend': 'fallback', 'model_name': 'ViT-B-32', 'prefer_pretrained': True, 'pretrained_requested': True, 'pretrained_tag': None, 'checkpoint_path': None, 'checkpoint_exists': False, 'weights_loaded': False, 'weights_source': None, 'local_files_only': True, 'allow_download': False, 'fallback_used': True, 'random_initialization_used': False, 'load_error': 'pretrained vision requested, but no local checkpoint_path was configured'}`
- `vision_pretrained_missing`: vision pretrained weights are required but not loaded. `{'requested_backend': 'clip', 'resolved_backend': 'fallback', 'model_name': 'ViT-B-32', 'prefer_pretrained': True, 'pretrained_requested': True, 'pretrained_tag': None, 'checkpoint_path': None, 'checkpoint_exists': False, 'weights_loaded': False, 'weights_source': None, 'local_files_only': True, 'allow_download': False, 'fallback_used': True, 'random_initialization_used': False, 'load_error': 'pretrained vision requested, but no local checkpoint_path was configured'}`
- `text_fallback_backbone`: text backbone is using fallback features. `{'requested_backend': 'transformers', 'resolved_backend': 'hashing', 'model_name': 'microsoft/deberta-v3-base', 'checkpoint_path': None, 'checkpoint_exists': False, 'weights_loaded': False, 'weights_source': None, 'local_files_only': True, 'allow_download': False, 'fallback_used': True, 'load_error': "OSError: We couldn't connect to 'https://huggingface.co' to load this file, couldn't find it in the cached files and it looks like microsoft/deberta-v3-base is not the path to a directory containing a file named config.json.\nCheckout your internet connection or see how to run the library in offline mode at '"}`
- `text_pretrained_missing`: text pretrained weights are required but not loaded. `{'requested_backend': 'transformers', 'resolved_backend': 'hashing', 'model_name': 'microsoft/deberta-v3-base', 'checkpoint_path': None, 'checkpoint_exists': False, 'weights_loaded': False, 'weights_source': None, 'local_files_only': True, 'allow_download': False, 'fallback_used': True, 'load_error': "OSError: We couldn't connect to 'https://huggingface.co' to load this file, couldn't find it in the cached files and it looks like microsoft/deberta-v3-base is not the path to a directory containing a file named config.json.\nCheckout your internet connection or see how to run the library in offline mode at '"}`
- `metric_contract_blocked`: Metric contract is not fully implementable under declared formal metric policy. `{'missing_capabilities': ['current structured evaluator consumes rendered tactic.rhetorical labels', 'validation-selected sigmoid threshold path for tactic logits is not yet exposed']}`

## Main-experiment acceptance decision
Main experiment results are valid only after `main_experiment` strict preflight passes.

## Required next actions
- Provide local pretrained vision/text checkpoints in `configs/config.yaml` and rerun strict preflight.
- Implement logits-only validation-threshold decoding for `tactic_rhetorical` formal metrics.
