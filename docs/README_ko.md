# 모듈형 유해 밈 해석 프레임워크

유해 밈 해석을 단순한 이진 탐지를 넘어 5개의 명시적 단계로 분해하는 모듈형 연구 프레임워크이다.

1. Internal Evidence Extractor
2. External Knowledge Acquisition
3. Knowledge Relevance Filter / Verifier
4. Evidence Fusion & Reasoning
5. Structured Interpretation Head

이 프레임워크는 **연구 프로토타이핑**, **ablation study**, **stage-wise debugging**, 그리고 **향후 확장**을 위해 설계되었다.  
또한 monolithic end-to-end 접근보다 유해 밈 해석을 더 **제어 가능하고**, **해석 가능하며**, **증거 기반적(evidence-grounded)** 으로 만드는 데 초점을 둔다.

---

## 1. 프로젝트 목표

유해 밈 이해는 단순한 이진 분류 문제가 아니다.  
하나의 밈을 이해하기 위해서는 다음이 필요할 수 있다.

- 이미지에 대한 시각적 이해
- OCR 텍스트에 대한 의미적 이해
- 국소적 심볼, 인물, 로고, 국기, 또는 템플릿 단서의 인식
- 멀티모달 불일치(multimodal incongruity) 또는 아이러니의 이해
- 외부의 문화적, 정치적, 또는 사건 관련 지식의 검색
- 검색된 지식이 실제로 관련성이 있고 뒷받침 근거가 되는지에 대한 검증
- 다음 항목들에 대한 구조화된 해석:
  - harmfulness
  - target
  - intent
  - tactic
  - supporting evidence
  - rationale

이 저장소는 이러한 작업 흐름을 지원하기 위한 모듈형 파이프라인을 구현한다.

---

## 2. 현재 데이터 레이아웃

이 프로젝트는 현재 다음과 같은 데이터셋 구조를 가정한다.

```text
.
├── dataset
│   └── V1
│       ├── covid_img+text
│       │   ├── img
│       │   └── txt
│       ├── facebook_img+text
│       │   ├── img
│       │   └── txt
│       ├── memotion_img+text
│       │   ├── img
│       │   └── txt
│       ├── political_img+text
│       │   ├── img
│       │   └── txt
├── result
├── module
├── outputs
│   ├── harm_c
│   │   └── harm_c_annotation.jsonl
│   ├── harm_p
│   ├── memotion
│   └── facebook
└── tool
    └── annotation
```

논리적 데이터셋 매핑:

- covid_img+text → harm_c
- political_img+text → harm_p
- memotion_img+text → memotion
- facebook_img+text → facebook

각 샘플은 다음을 포함할 수 있다.

- 밈 이미지
- OCR 텍스트
- raw label
- 다음을 포함하는 선택적 구조화 annotation:
  - target
  - intent
  - tactic
  - evidence
  - confidence metadata

## 3. 프레임워크 개요

이 프레임워크는 5단계 모듈형 파이프라인으로 구성된다.

### Stage A. Internal Evidence Extractor

직접 관찰 가능한 밈 콘텐츠로부터 구조화된 internal evidence bank를 구축한다.

주요 구성 요소:

- Global Visual Encoder
- Text Semantic Encoder
- Local Object / Symbol Extractor
- Cross-modal Incongruity Analyzer

출력:

- internal evidence tokens
- global visual token
- global text token
- ROI / symbol metadata
- incongruity token
- auxiliary internal scores

### Stage B. External Knowledge Acquisition

검색 지향적 query를 구성하고 외부 지식 후보를 수집한다.

주요 구성 요소:

- Query Constructor
- Entity / Concept Linking
- Hybrid Retrieval
- Context Augmentation Generator

출력:

- query bundle
- linked nodes
- retrieved knowledge candidates
- candidate knowledge tokens
- short context hypotheses

### Stage C. Knowledge Relevance Filter / Verifier

노이즈가 많은 검색 지식을 필터링하고, 추론에 사용할 검증된 증거만 남긴다.

주요 구성 요소:

- Evidence-aware Relevance Scorer
- Support / Contradiction Verifier
- Credibility / Temporal / Cultural Validator
- Redundancy Reduction

출력:

- verified knowledge bank
- verified knowledge tokens
- relevance/support/validity scores
- final filtered evidence set

### Stage D. Evidence Fusion & Reasoning

내부 증거와 검증된 외부 지식을 task-aware reasoning을 통해 융합한다.

주요 구성 요소:

- Internal Evidence Aggregator
- Knowledge-conditioned Cross-attention
- Gating
- Task-aware Reasoning Heads

출력:

- shared reasoning state
- target latent
- intent latent
- tactic latent
- fused token memory
- reasoning metadata

### Stage E. Structured Interpretation Head

최종 구조화 출력을 생성한다.

주요 구성 요소:

- Harmfulness Head
- Target Head
- Intent Head
- Tactic Head
- Evidence Attribution Layer
- Rationale Generator

출력:

- harmfulness
- target
- intent
- tactic
- supporting evidence
- rationale

## 4. 설계 원칙

이 코드베이스는 다음과 같은 중요한 설계 원칙을 따른다.

### 모듈성 중심 설계

각 stage는 독립적으로 실행 가능하고, inspect 가능하며, 교체 가능해야 한다.

### 연구 우선 구현

목표는 단순한 프로덕션 추론이 아니라 다음도 지원하는 것이다.

- stage-wise debugging
- ablation study
- intermediate evidence inspection
- structured output 평가

### Graceful fallback behavior

무거운 backbone을 사용할 수 없더라도, fallback logic으로 코드가 계속 실행되어야 한다.

### Shared hidden space

가능한 경우 stage 출력은 공유 hidden dimension(기본값: 256)을 통해 정렬되어야 한다.

### 자유 형식 설명보다 구조화 출력 우선

이 프레임워크는 다음을 우선한다.

- structured label prediction
- evidence attribution
- grounded rationale

그리고 그 다음에 더 자유로운 explanation generation을 고려한다.

## 5. 계획된 / 기대되는 프로젝트 구조

의도된 저장소 레이아웃은 다음과 같다.

```text
.
├── configs
├── dataset
├── module
│   ├── backbones
│   ├── stage_a
│   ├── stage_b
│   ├── stage_c
│   ├── stage_d
│   ├── stage_e
│   └── pipeline
├── scripts
├── utils
├── result
└── tests
```

주요 구현 영역:

### dataset/

- dataset loading
- sample mapping
- collate functions

### module/backbones/

- CLIP, text encoder, detector, retriever, cross-encoder, generator를 위한 wrapper와 adapter

### module/stage_a/ ~ module/stage_e/

- 5개 파이프라인 stage의 핵심 구현

### module/pipeline/

- 전체 파이프라인 모델, runner, inference logic

### scripts/

- stage-wise 또는 end-to-end 실행을 위한 CLI entry point

### utils/

- 재사용 가능한 helper function

### tests/

- smoke test 및 최소 stage test

### result/

- stage 출력, 최종 예측, 분석 artifact

## 6. 기대 입력과 출력

### 입력 샘플

최소한 시스템은 다음과 같은 통합 sample 구조를 지원해야 한다.

```python
{
    "sample_id": str,
    "dataset_name": str,
    "image_path": str,
    "ocr_text_full": str,
    "raw_label": int | None,
    "annotation": dict | None,
}
```

### 최종 출력

최종 구조화 출력은 다음과 같아야 한다.

```python
{
    "harmfulness": {
        "label": "harmful",
        "score": 0.92
    },
    "target": {
        "presence": "implicit",
        "granularity": "organization",
        "attributes": ["political_ideology"],
        "label": "Brexit에 관련된 정치 행위자",
        "score": 0.84
    },
    "intent": {
        "primary": "ridicule_mockery",
        "stance": "hostile",
        "secondary": ["political_criticism"],
        "background_knowledge_needed": True,
        "score": 0.80
    },
    "tactic": {
        "rhetorical": ["sarcasm_irony"],
        "multimodal_relation": "complementary",
        "structural": ["template_reuse", "panel_contrast"],
        "score": 0.87
    },
    "supporting_evidence": {
        "internal": [...],
        "external": [...]
    },
    "rationale": "..."
}
```

## 7. 개발 전략

구현은 두 단계로 진행하는 것을 의도한다.

### Phase 1

다음에 집중한다.

- 프로젝트 구조
- dataset loading
- stage schemas
- 실행 가능한 forward logic이 있는 stage modules
- CLI scripts
- result saving
- fallback behavior

목표:  
일부 구성 요소가 단순화되어 있더라도 end-to-end로 실행되는 전체 파이프라인.

### Phase 2

다음에 집중한다.

- 더 강력한 backbone과 adapter
- 개선된 retrieval과 linking
- 더 강력한 verification logic
- 더 나은 gated fusion
- 더 강력한 structured head
- 더 나은 evaluation과 analysis

목표:  
실행 가능한 scaffold를 더 현실적인 연구 구현으로 업그레이드하는 것.

## 8. 결과 저장

이 프레임워크는 debugging과 analysis를 위해 stage별로 출력을 저장해야 한다.

예:

- result/stage_a/internal_evidence.jsonl
- result/stage_b/knowledge_candidates.jsonl
- result/stage_c/verified_knowledge.jsonl
- result/stage_d/fusion_outputs.pt
- result/stage_e/final_predictions.jsonl

선택적인 per-sample export와 analysis-friendly summary도 강하게 권장된다.

## 9. 평가 목표

이 프레임워크는 최소한 다음 평가 방향을 지원해야 한다.

### Harmfulness

- AUROC
- Macro-F1
- Binary F1

### Target / Intent / Tactic

- Macro-F1
- 필요 시 Micro-F1
- structured output에 대한 field-specific accuracy

### Evidence attribution

- Precision@k
- Recall@k

### Rationale

- groundedness / faithfulness 우선
- text quality는 그 다음

## 10. 코딩 기대사항

이 저장소는 다음과 같아야 한다.

- Python 3.10+
- PyTorch 기반
- 필요한 경우 HuggingFace 호환
- modular
- readable
- inspectable
- easy to extend

다음을 선호하라.

- explicit code
- stable stage interfaces
- practical한 경우 schema에 dataclass 사용
- 누락된 optional dependency를 graceful하게 처리
- hardcoded path보다 configuration-driven behavior 사용

다음은 피하라.

- 거대한 monolithic script
- 강하게 결합된 stage 구현
- 절대경로에 대한 숨겨진 가정
- proprietary API 의존

## 11. 기여자 / Codex를 위한 가이드

이 저장소를 구현하거나 수정하고 있다면:

- 정말 필요한 경우가 아니라면 모든 것을 처음부터 다시 작성하지 마라.
- 5-stage modular structure를 유지하라.
- 각 stage를 독립적으로 테스트 가능하게 유지하라.
- 대규모 파괴적 리팩토링보다 기존 파일 개선을 선호하라.
- 논리적인 단계로 작고 검증 가능한 commit을 만들어라.
- result saving과 CLI 사용성을 유지하라.
- 무거운 구성 요소를 완전히 사용할 수 없는 경우 다음을 추가하라:
  - 깔끔한 adapter
  - fallback path
  - 명확한 TODO comment

우선순위:

1. dataset loader
2. stage schemas
3. stage module forward logic
4. pipeline runner
5. scripts
6. result export
7. evaluation helpers
8. deeper modeling improvements

## 12. 초기 TODO 체크리스트

- [ ] unified dataset loader 구현
- [ ] stage schemas 추가
- [ ] backbone/adapters 추가
- [ ] Stage A 구현
- [ ] Stage B 구현
- [ ] Stage C 구현
- [ ] Stage D 구현
- [ ] Stage E 구현
- [ ] pipeline runner 추가
- [ ] scripts 추가
- [ ] result saving 추가
- [ ] tests 추가
- [ ] evaluation utilities 추가

## 13. 장기 확장 아이디어

잠재적인 향후 확장:

- OCR box-level grounding
- 더 강력한 open-vocabulary detection
- 실제 FAISS + BM25 multi-source retrieval
- meme template knowledge base
- task-aware verified knowledge bank split
- constrained seq2seq rationale generation
- 더 강력한 evidence faithfulness evaluation
- reasoning을 위한 teacher-student distillation
- multilingual meme interpretation support

## 14. 저장소 상태

이 저장소는 현재 활발히 구현 중인 모듈형 연구 프로토타입이다.

의도된 최종 결과는, harmful meme interpretation이 다음과 같은 프레임워크이다.

- structured
- evidence-grounded
- stage-wise debuggable
- retrieval-aware
- target / intent / tactic 수준에서 설명 가능한

즉, 단순히 하나의 harmful / non-harmful score만 출력하는 시스템이 아니라는 뜻이다.
