# Paper Repository

이 저장소는 개인적으로 흥미 있게 읽은 논문들을 따라 구현해보며 학습한 내용을 정리한 공간입니다. 논문 내용을 실제 코드로 구현해보며 이해를 넓히는 데 초점을 두었습니다.


## 구현한 논문 목록

### 1. S-Walk: Accurate and Scalable Session-based Recommendation with Random Walks
- 추천 시스템 / 세션 기반 추천
- Random Walk 기반 세션 그래프 확산 방식 구현
- 논문 링크: [https://arxiv.org/abs/2108.05515](https://arxiv.org/abs/2108.05515)

### 2. How to Steer LLM Latents for Hallucination Detection?
- LLM 기반 Hallucination Detection
- Truthfulness Separator Vector (TSV) 개념 구현
- 주요 구성: TSV 레이어, TruthfulQA 실험, AUROC 평가
- 논문 링크: [https://arxiv.org/abs/2310.03684](https://arxiv.org/abs/2310.03684)

### 3. Attention-Guided Self-Reflection for Zero-Shot Hallucination Detection
- Zero-shot 환경에서의 hallucination 탐지
- Attention 기반 Self-Reflection 구조 구현
- 논문 링크: [https://arxiv.org/abs/2402.06679](https://arxiv.org/abs/2402.06679)


## 폴더 구조
```
paper/
├── s-walk/
├── tsv/
├── agser/
└── README.md
```


## 실행 환경 (공통)
- Python >= 3.9
- PyTorch >= 1.13
- Transformers
- Datasets (HuggingFace)
- Scikit-learn, SciPy

각 폴더에 있는 README.md 파일에 세부 사항과 실행 방법이 정리되어 있습니다.


## 참고
이 저장소는 어디까지나 개인적인 학습 및 실험 목적이며, 논문 이해를 돕기 위해 작성된 구현입니다.
코드가 완전하거나 최적화되어 있지 않을 수 있으며, 자유롭게 참고 또는 개선하셔도 됩니다.

