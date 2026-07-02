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
