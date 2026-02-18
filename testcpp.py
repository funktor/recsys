import numpy as np
import importlib
import pandas as pd
import os
import ml_32m_py
import joblib

importlib.reload(ml_32m_py)

def get_historical_user_features2(df2:pd.DataFrame, max_hist=20):
        user_ids = df2['userId'].to_numpy()
        movie_ids = df2['movieId'].to_numpy()
        ratings = df2['normalized_rating'].to_numpy().astype(np.float32)
        timestamps = df2['timestamp'].to_numpy()

        a, b  = ml_32m_py.py_get_historical_features(user_ids, movie_ids, timestamps, ratings, df2.shape[0], max_hist)

        df2["prev_movie_ids"] = a
        df2["prev_ratings"] = b

        return df2


df_train = joblib.load("df_train.pkl")
df2 = get_historical_user_features2(df_train)
print(df2[:100])