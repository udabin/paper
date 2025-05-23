# Attention-Guided Self-Reflection (AGSER)

이 폴더는 논문 [Attention-Guided Self-Reflection for Zero-Shot Hallucination Detection in Large Language Models](https://arxiv.org/abs/2402.06679)를 참고하여 AGSER 기법을 직접 구현한 코드입니다.

AGSER는 입력 쿼리 내 토큰의 중요도를 바탕으로 쿼리를 분리하고, 각각의 응답을 비교함으로써 LLM의 hallucination 가능성을 판단하는 기법입니다.


## 주요 구성

- **`AGSERConfig`**: 모델 하이퍼파라미터를 저장하는 데이터 클래스  
  - `k_ratio`: 주의 집중 토큰의 비율  
  - `lambda_balance`: hallucination score 조정을 위한 가중치  
  - `attention_type`: attention 계산 기준 (mean, mid, first, last 등)  
  - `max_length`, `temperature`, `top_p`: 응답 생성 제어  

- **`AGSER`**: 전체 AGSER 파이프라인을 수행하는 핵심 클래스  
  - `get_token_contributions(query)`: 쿼리에 대한 각 토큰의 기여도 추출 (hidden states 기반)  
  - `split_queries(query, scores)`: 토큰 중요도에 따라 쿼리를 집중 / 비집중 버전으로 나눔  
  - `generate_answer(query)`: 주어진 쿼리에 대해 응답 생성 (sampling 기반)  
  - `detect_hallucination(query)`: 원 응답, 집중/비집중 응답 간 Rouge-L 기반 비교 및 hallucination 점수 계산  

- **`MLXRougeScorer`**: Rouge-L 유사도를 직접 계산하기 위한 간단한 LCS 기반 클래스


## 주요 파일
- `AGSER-gpt2.py`: 전체 알고리즘 실행 예제가 포함된 메인 스크립트


## 실행 예시
```bash
python AGSER-gpt2.py
```
출력 예시:
```
Query: Who is the author of the book The Testament, what year was it published?
Hallucination Score: -0.1208
Original Answer: John Grisham wrote The Testament, published in 1999.
Attentive Answer: John Grisham is the author of The Testament.
Non-Attentive Answer: It was published in 1999.
```


## 핵심 아이디어
1. 입력 쿼리에 대해 LLM의 hidden state 기반 attention 기여도 계산
2. 상위 `k_ratio` 비율의 토큰으로 attentive query 생성
3. 나머지 토큰으로 non-attentive query 생성
4. 세 쿼리(original, att, non-att)에 대해 각각 응답 생성
5. Rouge-L 유사도로 비교 → hallucination score 계산


## 실행 환경
- Python >= 3.9
- PyTorch
- Transformers
- NumPy


## 참고
- 현재는 `GPT-2` 모델을 기준으로 테스트되며, 다양한 모델로 확장 가능
- 본 구현은 논문 내용을 실습하기 위한 개인용 코드이며, 실제 논문에서 사용하는 학습 기반 탐지 모델과는 다를 수 있음


## 원본 구현 출처
- 이 코드는 아래 공개된 GitHub 저장소를 바탕으로 작성되었으며 일부 구조 개선 및 모델이 추가되었습니다.
- 원본 구현: https://github.com/sanowl/AGSER
- 코드의 저작권은 원 저자에게 있으며, 본 저장소는 비상업적 학습 및 연구 목적으로 작성되었습니다.

