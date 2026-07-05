# Meme Detection Framework 요약

이 저장소는 유해 밈 해석을 위한 5단계 모듈형 파이프라인과 논문 실험용 러너를 포함합니다.

## 핵심 실행 명령

```bash
python scripts/run.py train --dataset harm_c --experiment ours_full --config configs/config.yaml --seed 42 --device cpu
python scripts/run.py baseline --dataset harm_c --baseline text_only_encoder --config configs/config.yaml --seed 42 --device cpu
python scripts/run.py stage --dataset harm_c --until stage_e --limit 5 --config configs/config.yaml --device cpu
python scripts/run.py audit --run-root result/predictions/harm_c/ours_full/42 --write-report --strict --require-nonempty-metrics
```

## 주요 코드 구조

```text
module/internal_evidence_extractor.py       # Stage A
module/external_knowledge_acquisition.py    # Stage B
module/knowledge_filter_verifier.py         # Stage C
module/evidence_fusion_reasoning.py         # Stage D
module/structured_interpretation_head.py    # Stage E
module/baseline.py                          # 단순 baseline 모델
module/losses.py                            # 구조화 loss
module/runner.py                            # 전체 파이프라인과 artifact 저장
module/backbone/                            # vision/text/retrieval/generation adapter
```

정규화 라벨 관련 코드는 `dataset/labels.py`에 있으며, 원본 데이터 경로 규칙은 유지됩니다.

```text
dataset/source
dataset/annotation
dataset/annotation_normalized
```

## 설정 파일

주 설정은 `configs/config.yaml`입니다. 라벨 vocab과 annotation normalization 설정은 별도 파일로 유지합니다.

```text
configs/config.yaml
configs/label_vocab.yaml
configs/annotation_normalization.yaml
```

## 출력 위치

- 단계별 artifact: `result/stage_a/` ~ `result/stage_e/`
- 예측 결과: `result/predictions/{dataset}/{model}/{seed}/`
- metric: `result/metrics/`
- 분석 결과: `result/analysis/`

## 실험 프로토콜 실행

논문용 메인 실험 전에 Experiment 0 preflight를 먼저 실행합니다.

```bash
python scripts/run.py preflight --profile smoke --config configs/config.yaml --dataset harm_c harm_p facebook memotion --seed 42 --label-set clean --device cpu --write-report
python scripts/run.py preflight --profile main_experiment --config configs/config.yaml --dataset harm_c harm_p facebook memotion --seed 42 --label-set clean --device cpu --write-report --strict
```

`smoke` preflight는 오프라인 구조 점검이며 fallback encoder 사용 시에도 경고와 함께 통과할 수 있습니다. `main_experiment` preflight는 논문용 실험 가능 여부를 판단하는 엄격한 게이트입니다. OpenCLIP/HuggingFace backend가 import되는 것만으로 pretrained checkpoint가 로드된 것으로 보지 않습니다. 실제 checkpoint 로드 여부(`weights_loaded`), fallback 사용 여부, random initialization 여부를 artifact에 기록합니다.

## 로컬 pretrained asset 준비

논문용 main experiment는 실제 로컬 pretrained asset이 필요합니다. Smoke fallback encoder는 개발용이며 paper-valid 결과로 보지 않습니다.

권장 순서:

```bash
python scripts/run.py assets init-layout --config configs/config.yaml
# 아래 경로에 호환되는 checkpoint/model snapshot을 직접 배치
python scripts/run.py assets verify --config configs/config.yaml --profile main_experiment --write-manifests --strict
python scripts/run.py preflight --profile main_experiment --config configs/config.yaml --dataset harm_c harm_p facebook memotion --seed 42 --label-set clean --device cpu --write-report --strict
```

필수 경로:

```text
assets/pretrained/vision/open_clip_vit_b_32/checkpoint.pt
assets/pretrained/text/deberta_v3_base/
```

텍스트 디렉터리는 `config.json`, tokenizer 파일, 그리고 `model.safetensors` 또는 `pytorch_model.bin`을 포함해야 합니다. 모델 weight 파일은 Git에 커밋하지 않습니다. `asset_manifest.json`과 `.gitkeep`만 추적 대상입니다.

DeBERTa-v3 텍스트 snapshot은 main experiment에서 slow SentencePiece tokenizer 경로를 명시적으로 사용합니다. `configs/config.yaml`의 `backbone.text.tokenizer_use_fast=false`, `tokenizer_backend_policy=sentencepiece_slow`가 기준이며, active environment에 다음 dependency가 필요합니다.

```bash
python -m pip install sentencepiece
```

Strict main preflight는 파일 존재 여부나 SHA-256만 확인하지 않습니다. Vision checkpoint가 설정된 OpenCLIP architecture와 실제로 호환되어야 하며, `checkpoint_compatibility_verified=true`, `shape_mismatch_count=0`, 그리고 manual loading 시 `matched_parameter_ratio>=0.99`가 필요합니다. Text는 `tokenizer_loaded=true`, `weights_loaded=true`, `sentencepiece_available=true`가 필요합니다. Random initialization, fallback embedding, shape mismatch, 낮은 key coverage, missing SentencePiece는 논문용 pretrained 상태로 인정하지 않습니다.

주요 preflight 출력:

```text
result/preflight/smoke/
result/preflight/main_experiment/
```

메인 실험 결과는 `main_experiment` strict preflight가 통과한 뒤에만 유효한 것으로 간주합니다.

논문용 반복 실험은 통합 CLI의 suite 명령을 권장합니다.

```bash
python scripts/run.py suite --suite core_smoke --config configs/config.yaml --device cpu --dry-run
python scripts/run.py suite --suite core_smoke --config configs/config.yaml --device cpu --disable-tqdm --audit-after-run --strict --require-nonempty-metrics
```

suite 실행은 dataset/seed별 split 파일을 하나로 고정하고, 각 run마다 `run_manifest.json`을 저장합니다.

```text
result/splits/{dataset}/seed_{seed}.json
result/predictions/{dataset}/{model}/{seed}/run_manifest.json
result/experiment_suites/{suite_name}/suite_manifest.json
```

ablation 의미와 감사(audit) 기준은 `experiments/ablation_configs.py`의 `AblationContract`와 `docs/EXPERIMENT_PROTOCOL.md`에 정리되어 있습니다.

## Formal `tactic_rhetorical` metric

논문용 `tactic_rhetorical` metric은 Stage E의 trainable tactic logits만 사용합니다.

- 확률 변환: sigmoid
- threshold: validation macro-F1 기준으로 dataset/run/seed별 1회 선택
- test 평가: validation에서 선택한 threshold를 고정 적용
- `none`: non-none tactic이 하나도 선택되지 않을 때만 fallback으로 사용
- 제외 필드: `tactic.rhetorical`, `tactic.rhetorical_labels`, heuristic rhetorical cue, Stage A cue, rationale text

렌더링된 tactic label은 rationale와 case-study를 위한 설명용 diagnostic이며, paper-facing formal 성능 metric이 아닙니다. 관련 artifact는 다음 위치에 저장됩니다.

```text
result/predictions/{dataset}/{model}/{seed}/tactic_rhetorical_decoding.json
```
