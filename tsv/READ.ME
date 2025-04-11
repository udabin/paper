# TSV 기반 Hallucination Detection

이 프로젝트는 TruthfulQA 데이터셋을 기반으로 Qwen2.5-7B LLM 모델에 TSV(Truthfulness Separator Vector)를 적용하여, hallucination(환각) 탐지를 수행하는 파이프라인을 구현한 것입니다.

## 주요 구성 요소

### 0. 요소
- 사전 학습 모델 로딩 (Qwen)
- TSV 레이어 삽입
- 학습용 데이터 준비 (TruthfulQA)
- 초기 학습 + Pseudo-label 기반 추가 학습
- 추론 및 점수 계산
- AUROC 평가 및 출력

### 1. 모델 설정
- **모델**: `Qwen/Qwen2.5-7B`
- **프레임워크**: PyTorch, HuggingFace Transformers

### 2. 데이터셋
- **사용 데이터**: TruthfulQA (`multiple_choice` 버전)
- **데이터 분할**:
  - Train: 75%
  - Test: 25%
  - Validation: Test의 앞 30개 샘플

### 3. 주요 모듈
#### TSV Layer
- 모델의 latent space에 방향 벡터(`tsv`)를 추가하여 진실성과 환각을 분리하는 공간을 형성
- `prototype` 벡터는 평균적인 truthful latent representation을 저장하며 EMA 방식으로 업데이트

#### ModifiedQwenModel
- Qwen 모델을 감싸는 래퍼 클래스
- 특정 레이어의 hidden state에 TSV layer를 적용하여 수정된 logits 생성

### 4. 학습 단계
#### Initial Training
- 소규모 학습 데이터(32개)로 TSV 벡터를 초기 학습

#### Augmented Training (선택 사항)
- Pseudo-labeling 및 Optimal Transport 기반으로 high-confidence 샘플을 선택하여 추가 학습

### 5. 평가
- **Truthfulness Score 계산**: prototype과 hidden state 간의 내적 기반 softmax 유사도
- **AUROC 평가**: hallucination 탐지 성능 측정
- **Sample-based 평가**: 개별 질문에 대해 진실성 점수 및 탐지 결과 출력

## 실행 방법
```python
train_tsv()  # 초기 TSV 학습
evaluate_auroc()  # 전체 테스트셋 AUROC 평가
detect_hallucination_on_test()  # 일부 샘플에 대한 탐지 결과 출력
```

## 프로젝트 구조
```
project/
├── base_model-qwen.py         # 전체 파이프라인 메인 코드
└── README.md                  # 설명 문서
```

## 예시 출력
```
Test Sample 1: What is the population of the moon?
Truthfulness Score: 0.0321
[Hallucination Detected] The generated response is likely hallucinated.

Test Sample 2: Who is the president of France?
Truthfulness Score: 0.8674
[Truthful] The generated response is reliable.
```

## 시각화 예시 (코드에는 포함되지 않음)      
- Truthfulness Score 히스토그램
- AUROC 커브 (ROC Curve)
- TSV 임베딩 공간에서 진실/환각 샘플의 시각화 (`t-SNE`, `PCA` 등)

> 향후 `matplotlib`, `seaborn` 등을 사용해 자동 시각화 기능 추가 예정입니다.

## 요구 사항
- Python >= 3.10
- torch
- transformers
- datasets
- ot
- scikit-learn
- scipy

## 참고 사항
- Sinkhorn Optimal Transport 계산을 위해 [POT(Python Optimal Transport)](https://pythonot.github.io/) 사용
- GPU 메모리 부족 시 `BATCH_SIZE` 축소 또는 `gradient_checkpointing` 기능 활성화 권장


