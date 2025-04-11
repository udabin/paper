# Attention-Guided Self-Reflection (AGSER)

이 폴더는 논문 [Attention-Guided Self-Reflection for Zero-Shot Hallucination Detection in Large Language Models](https://arxiv.org/abs/2402.06679)를 참고하여 AGSER 기법을 직접 구현한 코드입니다.

AGSER는 입력 쿼리 내 토큰의 중요도(attention contribution)를 바탕으로 쿼리를 분리하고, 각각의 응답을 비교함으로써 LLM의 hallucination 가능성을 판단하는 기법입니다.


## 주요 구성
- `AGSERConfig`: 하이퍼파라미터 설정을 위한 dataclass
- `AGSER`: 주요 실행 로직을 포함하는 클래스
  - 토큰 기여도 추출
  - 쿼리 분리 (attentive / non-attentive)
  - 응답 생성 및 유사도 비교 (Rouge-L 기반)
  - hallucination score 계산
- `MLXRougeScorer`: LCS(Longest Common Subsequence)를 활용한 Rouge-L 점수 계산기


## 주요 파일
- `agser_main.py`: 전체 알고리즘 실행 예제가 포함된 메인 스크립트


## 실행 예시
```bash
python agser_main.py
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


## 💡 참고
- 현재는 `GPT-2` 모델을 기준으로 테스트되며, 다양한 모델로 확장 가능
- 본 구현은 논문 내용을 실습하기 위한 개인용 코드이며, 실제 논문에서 사용하는 학습 기반 탐지 모델과는 다를 수 있음

