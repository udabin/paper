from google.colab import drive
import pandas as pd


drive.mount('/content/gdrive/')
path = './gdrive/MyDrive/ml-32m/'


def preprocess(path, )

    movies_data = pd.read_csv(path + 'movies.csv')
    ratings_data = pd.read_csv(path + 'ratings.csv')
    tags_data = pd.read_csv(path + 'tags.csv')

    pre_data = pd.merge(ratings_data, tags_data, how='inner', on=['userId', 'movieId'])
    data = pd.merge(pre_data, movies_data, how='left', on='movieId')
    data = data.rename(columns={"timestamp_x": "rating_timestamp", "timestamp_y": "tag_timestamp"})
    
    # data.isnull().sum()
    
    data['rating_date'] = pd.to_datetime(data['rating_timestamp'], unit='s').dt.date
    data['tag_date'] = pd.to_datetime(data['tag_timestamp'], unit='s').dt.date
    
    # user 별 데이터 체크
    # data[data['userId'] == 162279][['userId', 'rating_date']]
    
    # 'userId'와 'rating_date'를 기준으로 그룹화하고 'movieId'를 리스트로 묶어줌
    result = data.groupby(['userId', 'rating_date'])['movieId'].apply(list).reset_index()
    
    # 각 'userId'별로 리스트로 묶어줌
    final_result = result.groupby('userId').apply(lambda x: x[['rating_date', 'movieId']].to_dict(orient='records')).reset_index(name='movie_list')
    
    # movie_list 컬럼을 explode를 사용해 각 dict를 개별 row로 분리
    exploded_result = final_result.explode('movie_list').reset_index(drop=True)
    
    # movie_list 컬럼을 다시 분리하여 'rating_date'와 'movieId'로 분리
    exploded_result['rating_date'] = exploded_result['movie_list'].apply(lambda x: x['rating_date'])
    exploded_result['movieId'] = exploded_result['movie_list'].apply(lambda x: x['movieId'])
    
    # movie_list 컬럼 삭제
    exploded_result = exploded_result.drop(columns=['movie_list'])
    
    # 각 'userId'별로 'rating_date'에 따른 'movieId' 리스트를 하나의 열로 묶어줌
    final_grouped_result = exploded_result.groupby('userId').apply(lambda x: x['movieId'].to_list()).reset_index(name='movie_list')

    return final_grouped_result

