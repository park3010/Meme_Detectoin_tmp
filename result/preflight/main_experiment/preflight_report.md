# Experiment 0 Preflight Report

## Profile and command
- Profile: `main_experiment`
- Strict: `True`
- Config: `configs/config.yaml`
- Config SHA256: `419969b1228bf19ec7760f7f108a60b69dd3998077b1317223ce97037061aeb4`
- Created: `2026-07-05T09:28:43.958463+00:00`

## Overall decision
**PASS_WITH_WARNINGS**

## Backbone readiness
- Vision main-ready: `True`
- Text main-ready: `True`
- Vision state: `{'requested_backend': 'clip', 'resolved_backend': 'open_clip', 'model_name': 'ViT-B-32', 'prefer_pretrained': True, 'pretrained_requested': True, 'pretrained_tag': None, 'checkpoint_path': 'assets/pretrained/vision/open_clip_vit_b_32/checkpoint.pt', 'resolved_path': '/home/sujin/psj2003/meme_detection/assets/pretrained/vision/open_clip_vit_b_32/checkpoint.pt', 'asset_mode': 'local_checkpoint', 'checkpoint_exists': True, 'checkpoint_sha256': '149d74555e1e5a8a500990dcd355f6b0fcbc677bc144ac0393a2e20ce9e298f7', 'checkpoint_format': 'open_clip_factory_local_path', 'checkpoint_compatibility_verified': True, 'checkpoint_model_name': 'ViT-B-32', 'checkpoint_parameter_key_count': 0, 'model_parameter_key_count': 302, 'matched_parameter_key_count': 302, 'matched_parameter_numel': 151277313, 'model_parameter_numel': 151277313, 'matched_parameter_ratio': 1.0, 'missing_key_count': 0, 'unexpected_key_count': 0, 'shape_mismatch_count': 0, 'compatibility_failure_reason': None, 'factory_local_load_error': None, 'weights_loaded': True, 'weights_source': 'local_checkpoint', 'local_files_only': True, 'allow_download': False, 'fallback_used': False, 'random_initialization_used': False, 'load_error': None}`
- Text state: `{'requested_backend': 'transformers', 'resolved_backend': 'transformers', 'model_name': 'microsoft/deberta-v3-base', 'checkpoint_path': 'assets/pretrained/text/deberta_v3_base', 'resolved_path': '/home/sujin/psj2003/meme_detection/assets/pretrained/text/deberta_v3_base', 'asset_mode': 'local_directory', 'checkpoint_exists': True, 'checkpoint_sha256': 'fcea072a96d9872c3b44d5b3514e7acb8ca02a9ab95933a7e1fc8724c21bb85c', 'weights_loaded': True, 'weights_source': 'local_directory', 'local_files_only': True, 'allow_download': False, 'tokenizer_use_fast': False, 'tokenizer_backend_policy': 'sentencepiece_slow', 'tokenizer_class': 'DebertaV2Tokenizer', 'tokenizer_loaded': True, 'sentencepiece_required': True, 'sentencepiece_available': True, 'fallback_used': False, 'load_error': None}`

## Pretrained asset audit
- Passed: `True`
- Artifact: `result/preflight/main_experiment/pretrained_asset_audit.json`
- Vision asset: `/home/sujin/psj2003/meme_detection/assets/pretrained/vision/open_clip_vit_b_32/checkpoint.pt`
- Text asset: `/home/sujin/psj2003/meme_detection/assets/pretrained/text/deberta_v3_base`

## Dataset / label eligibility
- Datasets: `['harm_c', 'harm_p', 'facebook', 'memotion']`
- Detailed CSV: `result/preflight/main_experiment/dataset_metric_eligibility.csv`

## Split integrity
- Split report: `result/preflight/main_experiment/split_integrity_report.json`

## Retrieval corpus readiness
- Usable corpora: `1`
- Provenance note: `fallback candidate != retrieved external knowledge; generated hypothesis != retrieved external knowledge`

## Metric contract
- Implementation status: `ready`
- Missing capabilities: `[]`

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
- None

## Main-experiment acceptance decision
Main experiment results are valid only after `main_experiment` strict preflight passes.

## Required next actions
- None
