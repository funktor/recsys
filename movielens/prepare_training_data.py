import pandas as pd
import os
import joblib

base_dir = os.path.abspath(os.path.dirname(__file__))

def split_movielens(path, out_dir):
    column_names = ['userId', 'movieId', 'rating', 'timestamp']
    df = pd.read_csv(path, sep=',', names=column_names, dtype={'userId':'int32', 'movieId':'int32', 'rating':float, 'timestamp':'int64'}, header=0)
    df.dropna(inplace=True)

    n_users = df['userId'].nunique()
    n_movies = df['movieId'].nunique()

    max_df_ts = df[['userId', 'timestamp']].groupby('userId', as_index=False).max()

    df_merged = df.merge(max_df_ts, on=['userId'], how='left', suffixes=('', '_max_ts'))

    train_val_df = df_merged[df_merged['timestamp'] < df_merged['timestamp_max_ts']][column_names]
    test_df = df_merged[df_merged['timestamp'] >= df_merged['timestamp_max_ts']][column_names]

    train_df = train_val_df.sample(frac=0.8, random_state=42)
    val_df = train_val_df.drop(train_df.index)

    train_df.to_csv(os.path.join(out_dir, "train.csv"), sep=",")
    test_df.to_csv(os.path.join(out_dir, "test.csv"), sep=",")
    val_df.to_csv(os.path.join(out_dir, "validation.csv"), sep=",")

    return n_users, n_movies

def ratings_only_datasets(root_dir='/Users/amondal/recsys'):
    out_dir = os.path.join(base_dir, 'training_data')
    os.makedirs(out_dir, exist_ok=True)

    dataset_path = os.path.join(root_dir, 'datasets', 'ml-32m', 'ratings.csv')
    n_users, n_movies = split_movielens(path=dataset_path, out_dir=out_dir)
    joblib.dump((n_users, n_movies), os.path.join(out_dir, 'num_users_movies.pkl'))