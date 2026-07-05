# Experiment 0 Preflight Report

## Profile and command
- Profile: `smoke`
- Strict: `False`
- Config: `configs/config.yaml`
- Config SHA256: `c16db4e969381322c620a74ee1f0eaafed4b5d498231945e2b05e73a5838c915`
- Created: `2026-07-05T08:20:04.547616+00:00`

## Overall decision
**PASS_WITH_WARNINGS**

## Backbone readiness
- Vision main-ready: `True`
- Text main-ready: `True`
- Vision state: `{'requested_backend': 'clip', 'resolved_backend': 'fallback', 'model_name': 'ViT-B-32', 'prefer_pretrained': True, 'pretrained_requested': True, 'pretrained_tag': None, 'checkpoint_path': 'assets/pretrained/vision/open_clip_vit_b_32/checkpoint.pt', 'resolved_path': '/home/sujin/psj2003/meme_detection/assets/pretrained/vision/open_clip_vit_b_32/checkpoint.pt', 'asset_mode': 'local_checkpoint', 'checkpoint_exists': False, 'checkpoint_sha256': None, 'weights_loaded': False, 'weights_source': None, 'local_files_only': True, 'allow_download': False, 'fallback_used': True, 'random_initialization_used': False, 'load_error': 'checkpoint_path does not exist: assets/pretrained/vision/open_clip_vit_b_32/checkpoint.pt'}`
- Text state: `{'requested_backend': 'transformers', 'resolved_backend': 'hashing', 'model_name': 'microsoft/deberta-v3-base', 'checkpoint_path': 'assets/pretrained/text/deberta_v3_base', 'resolved_path': '/home/sujin/psj2003/meme_detection/assets/pretrained/text/deberta_v3_base', 'asset_mode': 'local_directory', 'checkpoint_exists': True, 'checkpoint_sha256': '3b35f9e3619ef897f479671f7fda363a5e0e256bdfcd59d6ed66ab80f0e3964b', 'weights_loaded': False, 'weights_source': None, 'local_files_only': True, 'allow_download': False, 'fallback_used': True, 'load_error': "OSError: Can't load tokenizer for 'assets/pretrained/text/deberta_v3_base'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure 'assets/pretrained/text/deberta_v3_base' is the correct path to a directory conta"}`

## Pretrained asset audit
- Passed: `True`
- Artifact: `result/preflight/smoke/pretrained_asset_audit.json`
- Vision asset: `/home/sujin/psj2003/meme_detection/assets/pretrained/vision/open_clip_vit_b_32/checkpoint.pt`
- Text asset: `/home/sujin/psj2003/meme_detection/assets/pretrained/text/deberta_v3_base`

## Dataset / label eligibility
- Datasets: `['harm_c', 'harm_p', 'facebook', 'memotion']`
- Detailed CSV: `result/preflight/smoke/dataset_metric_eligibility.csv`

## Split integrity
- Split report: `result/preflight/smoke/split_integrity_report.json`

## Retrieval corpus readiness
- Usable corpora: `1`
- Provenance note: `fallback candidate != retrieved external knowledge; generated hypothesis != retrieved external knowledge`

## Metric contract
- Implementation status: `ready`
- Missing capabilities: `[]`

## Normalized annotation snapshot
- Snapshot artifact: `result/preflight/smoke/normalized_annotation_snapshot.json`

## Warnings
- `vision_fallback_backbone`: vision backbone is using fallback features. `{'requested_backend': 'clip', 'resolved_backend': 'fallback', 'model_name': 'ViT-B-32', 'prefer_pretrained': True, 'pretrained_requested': True, 'pretrained_tag': None, 'checkpoint_path': 'assets/pretrained/vision/open_clip_vit_b_32/checkpoint.pt', 'resolved_path': '/home/sujin/psj2003/meme_detection/assets/pretrained/vision/open_clip_vit_b_32/checkpoint.pt', 'asset_mode': 'local_checkpoint', 'checkpoint_exists': False, 'checkpoint_sha256': None, 'weights_loaded': False, 'weights_source': None, 'local_files_only': True, 'allow_download': False, 'fallback_used': True, 'random_initialization_used': False, 'load_error': 'checkpoint_path does not exist: assets/pretrained/vision/open_clip_vit_b_32/checkpoint.pt'}`
- `vision_checkpoint_missing`: vision checkpoint_path was configured but does not exist. `{'requested_backend': 'clip', 'resolved_backend': 'fallback', 'model_name': 'ViT-B-32', 'prefer_pretrained': True, 'pretrained_requested': True, 'pretrained_tag': None, 'checkpoint_path': 'assets/pretrained/vision/open_clip_vit_b_32/checkpoint.pt', 'resolved_path': '/home/sujin/psj2003/meme_detection/assets/pretrained/vision/open_clip_vit_b_32/checkpoint.pt', 'asset_mode': 'local_checkpoint', 'checkpoint_exists': False, 'checkpoint_sha256': None, 'weights_loaded': False, 'weights_source': None, 'local_files_only': True, 'allow_download': False, 'fallback_used': True, 'random_initialization_used': False, 'load_error': 'checkpoint_path does not exist: assets/pretrained/vision/open_clip_vit_b_32/checkpoint.pt'}`
- `text_fallback_backbone`: text backbone is using fallback features. `{'requested_backend': 'transformers', 'resolved_backend': 'hashing', 'model_name': 'microsoft/deberta-v3-base', 'checkpoint_path': 'assets/pretrained/text/deberta_v3_base', 'resolved_path': '/home/sujin/psj2003/meme_detection/assets/pretrained/text/deberta_v3_base', 'asset_mode': 'local_directory', 'checkpoint_exists': True, 'checkpoint_sha256': '3b35f9e3619ef897f479671f7fda363a5e0e256bdfcd59d6ed66ab80f0e3964b', 'weights_loaded': False, 'weights_source': None, 'local_files_only': True, 'allow_download': False, 'fallback_used': True, 'load_error': "OSError: Can't load tokenizer for 'assets/pretrained/text/deberta_v3_base'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure 'assets/pretrained/text/deberta_v3_base' is the correct path to a directory conta"}`
- `vision_checkpoint_missing`: Vision checkpoint file is missing. `{'path': '/home/sujin/psj2003/meme_detection/assets/pretrained/vision/open_clip_vit_b_32/checkpoint.pt'}`
- `text_required_files_missing`: Text model directory is incomplete. `{'path': '/home/sujin/psj2003/meme_detection/assets/pretrained/text/deberta_v3_base', 'missing': ['config.json', 'tokenizer_config.json', 'tokenizer.json|spm.model|vocab.json|vocab.txt', 'model.safetensors|pytorch_model.bin|sharded weights']}`
- `vision_runtime_weights_not_loaded`: vision runtime did not load pretrained weights. `{'requested_backend': 'clip', 'resolved_backend': 'fallback', 'model_name': 'ViT-B-32', 'prefer_pretrained': True, 'pretrained_requested': True, 'pretrained_tag': None, 'checkpoint_path': 'assets/pretrained/vision/open_clip_vit_b_32/checkpoint.pt', 'resolved_path': '/home/sujin/psj2003/meme_detection/assets/pretrained/vision/open_clip_vit_b_32/checkpoint.pt', 'asset_mode': 'local_checkpoint', 'checkpoint_exists': False, 'checkpoint_sha256': None, 'weights_loaded': False, 'weights_source': None, 'local_files_only': True, 'allow_download': False, 'fallback_used': True, 'random_initialization_used': False, 'load_error': 'checkpoint_path does not exist: assets/pretrained/vision/open_clip_vit_b_32/checkpoint.pt'}`
- `vision_runtime_fallback_used`: vision runtime used fallback features. `{'requested_backend': 'clip', 'resolved_backend': 'fallback', 'model_name': 'ViT-B-32', 'prefer_pretrained': True, 'pretrained_requested': True, 'pretrained_tag': None, 'checkpoint_path': 'assets/pretrained/vision/open_clip_vit_b_32/checkpoint.pt', 'resolved_path': '/home/sujin/psj2003/meme_detection/assets/pretrained/vision/open_clip_vit_b_32/checkpoint.pt', 'asset_mode': 'local_checkpoint', 'checkpoint_exists': False, 'checkpoint_sha256': None, 'weights_loaded': False, 'weights_source': None, 'local_files_only': True, 'allow_download': False, 'fallback_used': True, 'random_initialization_used': False, 'load_error': 'checkpoint_path does not exist: assets/pretrained/vision/open_clip_vit_b_32/checkpoint.pt'}`
- `text_runtime_weights_not_loaded`: text runtime did not load pretrained weights. `{'requested_backend': 'transformers', 'resolved_backend': 'hashing', 'model_name': 'microsoft/deberta-v3-base', 'checkpoint_path': 'assets/pretrained/text/deberta_v3_base', 'resolved_path': '/home/sujin/psj2003/meme_detection/assets/pretrained/text/deberta_v3_base', 'asset_mode': 'local_directory', 'checkpoint_exists': True, 'checkpoint_sha256': '3b35f9e3619ef897f479671f7fda363a5e0e256bdfcd59d6ed66ab80f0e3964b', 'weights_loaded': False, 'weights_source': None, 'local_files_only': True, 'allow_download': False, 'fallback_used': True, 'load_error': "OSError: Can't load tokenizer for 'assets/pretrained/text/deberta_v3_base'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure 'assets/pretrained/text/deberta_v3_base' is the correct path to a directory conta"}`
- `text_runtime_fallback_used`: text runtime used fallback features. `{'requested_backend': 'transformers', 'resolved_backend': 'hashing', 'model_name': 'microsoft/deberta-v3-base', 'checkpoint_path': 'assets/pretrained/text/deberta_v3_base', 'resolved_path': '/home/sujin/psj2003/meme_detection/assets/pretrained/text/deberta_v3_base', 'asset_mode': 'local_directory', 'checkpoint_exists': True, 'checkpoint_sha256': '3b35f9e3619ef897f479671f7fda363a5e0e256bdfcd59d6ed66ab80f0e3964b', 'weights_loaded': False, 'weights_source': None, 'local_files_only': True, 'allow_download': False, 'fallback_used': True, 'load_error': "OSError: Can't load tokenizer for 'assets/pretrained/text/deberta_v3_base'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure 'assets/pretrained/text/deberta_v3_base' is the correct path to a directory conta"}`
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
