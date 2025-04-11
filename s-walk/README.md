# S-Walk: Session-based Recommendation with Random Walks

이 폴더는 논문 [S-Walk: Accurate and Scalable Session-based Recommendation with Random Walks](https://arxiv.org/abs/2108.05515)의 구현을 개인적으로 실습한 코드입니다.


## 주요 내용
- 세션 기반 추천 시스템의 핵심 기법 중 하나인 **Random Walk 기반 사용자 세션 모델링** 구현
- `ratings.csv`, `movies.csv`, `tags.csv` 데이터를 통합해 **사용자-세션-아이템** 구조를 구성
- 사용자별로 날짜 기준의 시퀀스를 구성하여 향후 추천 알고리즘 학습에 적합한 형식으로 전처리


## 주요 파일
- `A01_Preprocess.py`: 원본 데이터를 불러와 사용자 세션 단위로 movieId 리스트를 구성하는 전처리 코드


## 데이터 구성
데이터는 `MovieLens`의 `ml-32m` 버전 기준이며, 다음과 같은 파일이 필요합니다:
- `ratings.csv`
- `movies.csv`
- `tags.csv`

Colab에서는 다음 경로로 연결되어 있다고 가정합니다:
```
./gdrive/MyDrive/ml-32m/
```


## 전처리 과정 요약
1. `ratings`, `tags`, `movies`를 merge
2. `userId`, `rating_date` 단위로 영화 시청 내역을 그룹화
3. 사용자별로 날짜 기반의 영화 시퀀스를 dict 형식으로 구성
4. 이를 row별로 explode하여 최종 `userId`별 시퀀스 데이터 생성

---

## 출력 예시
| userId | movie_list |
|--------|------------|
| 1 | [1, 34, 82, 9] |
| 2 | [12, 17, 98] |

이런 형식으로 사용자별 시청 이력을 리스트로 정리합니다.


## 참고
- 현재 업로드 된 코드는 추천 알고리즘 적용을 위한 **전처리용 스크립트**입니다.
- 본 논문에서 제시한 Random Walk 기반 모델 학습은 이후 단계에서 이어질 수 있습니다.


## 실행 환경
- Python 3.9 이상
- pandas
- Google Colab 환경 (Google Drive mount)

