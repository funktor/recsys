import pandas as pd
import os
import numpy as np
import joblib
import importlib
import ml_32m_py
import numpy as np
import random
import shutil
import pyarrow
from pathlib import Path
from google.cloud import storage
from google.cloud.storage import Client, transfer_manager

def remove_stop(x):
    stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
    out = []
    for y in x:
        if len(y) > 0 and y not in stopwords:
            out += [y]
    return out

def flatten_lists(x):
    out = set()
    for y in x:
        out.update(y.split(" "))
    out = list(out)
    return out

def get_ml_32m_dataframe(path:str):
    ratings_path = os.path.join(path, 'ratings.csv')
    movies_path = os.path.join(path, 'movies.csv')
    tags_path = os.path.join(path, 'tags.csv')

    rating_column_names = ['userId', 'movieId', 'rating', 'timestamp']
    movies_column_names = ['movieId', 'title', 'genres']
    tags_column_names = ['userId', 'movieId', 'tag', 'timestamp']

    df_ratings = pd.read_csv(ratings_path, sep=',', names=rating_column_names, dtype={'userId':'int32', 'movieId':'int32', 'rating':float, 'timestamp':'int64'}, header=0)
    df_movies = pd.read_csv(movies_path, sep=',', names=movies_column_names, dtype={'movieId':'int32', 'title':'object', 'genres':'object'}, header=0)
    df_tags = pd.read_csv(tags_path, sep=',', names=tags_column_names, dtype={'userId':'int32', 'movieId':'int32', 'tag':'object', 'timestamp':'int64'}, header=0)

    df_ratings.dropna(inplace=True, subset=['userId', 'movieId', 'rating'])
    df_movies.dropna(inplace=True, subset=['movieId', 'title', 'genres'])
    df_tags.dropna(inplace=True, subset=['userId', 'movieId', 'tag'])
    df_tags.drop(columns=["userId","timestamp"], inplace=True)

    # Extract movie genres
    df_movies['genres'] = df_movies['genres'].apply(lambda x: x.lower().split('|'))

    df_movies['movie_year'] = df_movies['title'].str.extract(r'\((\d{4})\)').fillna("2025").astype('int')

    df_movies['title'] = df_movies['title'].str.replace(r'\((\d{4})\)', '', regex=True)
    df_movies['title'] = df_movies['title'].str.replace(r'[^a-zA-Z0-9\s]+', '', regex=True)
    df_movies['title'] = df_movies['title'].apply(lambda x: x.strip().lower().split(" "))
    df_movies['title'] = df_movies['title'].apply(lambda x: remove_stop(x))

    df_tags['tag'] = df_tags['tag'].str.replace(r'[^a-zA-Z0-9\s]+', '', regex=True)
    df_tags['tag'] = df_tags['tag'].apply(lambda x: x.strip().lower())
    df_tags = df_tags.groupby("movieId").agg(set).reset_index()
    df_tags['tag'] = df_tags['tag'].apply(list)
    df_tags['tag'] = df_tags['tag'].apply(lambda x: flatten_lists(x))
    df_tags['tag'] = df_tags['tag'].apply(lambda x: remove_stop(x))
    df_tags['tag'] = df_tags['tag'].astype("object")

    df_movies = df_movies.merge(df_tags, on=['movieId'], how='left')
    df_movies["tag"] = df_movies["tag"].fillna({i: [""] for i in df_movies.index})
    df_movies["description"] = df_movies["title"] + df_movies["tag"]
    df_movies.drop(columns=["tag"], inplace=True)
    df_movies.drop(columns=["title"], inplace=True)

    return df_ratings, df_movies

def normalize_ratings(df:pd.DataFrame):
    df2 = df[["userId", "rating"]].groupby(by=["userId"]).agg(mean_user_rating=('rating', 'mean'), std_user_rating=('rating', 'std'))
    df = df.merge(df2, on=["userId"], how="inner")
    df["normalized_rating"] = (df["rating"] - df["mean_user_rating"])/df["std_user_rating"]
    df["normalized_rating"] = df["normalized_rating"].fillna(df["rating"])
    df.drop(columns=["rating"], inplace=True)
    return df

def split_train_test(df:pd.DataFrame, min_rated=10, test_ratio=0.8, val_ratio=0.8):
    print("Splitting data into train test and validation...")
    # Split data into training, testing and validation
    df = df.sort_values(by='timestamp')
    df2 = df[["userId", "movieId"]].groupby(by=["userId"]).agg(list).reset_index()

    # Filter all user_ids who have rated more than 'min_rated' movies
    df2 = df2[df2.movieId.apply(len) > min_rated]
    df = df.merge(df2, on=["userId"], how="inner", suffixes=("", "_right"))
    df.drop(columns=['movieId_right'], inplace=True)

    n = df.shape[0]
    m = int(test_ratio*n)

    df_train_val = df[:m]
    df_test = df[m:]

    k = int(val_ratio*m)
    df_train = df_train_val[:k]
    df_val = df_train_val[k:]

    return df_train, df_val, df_test


def transform(x, vocab):
    if isinstance(x, list):
        out = []
        for y in x:
            out += [vocab[y]] if y in vocab else [0]
        return out
    else:
        return vocab[x] if x in vocab else 0
    
def categorical_encoding(df:pd.DataFrame, col:str, max_vocab_size=1000):
    all_vals = df[col].tolist()
    unique_vals = {}

    if len(all_vals) > 0 and isinstance(all_vals[0], list):
        for v in all_vals:
            for x in v:
                if x not in unique_vals:
                    unique_vals[x] = 0
                unique_vals[x] += 1
    else:
        for x in all_vals:
            if x not in unique_vals:
                unique_vals[x] = 0
            unique_vals[x] += 1
    
    unique_vals = sorted(unique_vals.items(), key=lambda item: item[1], reverse=True)
    unique_vals = dict(unique_vals[:min(max_vocab_size, len(unique_vals))])
    unique_vals = sorted(unique_vals.keys())
    vocab = {unique_vals[i] : i+1 for i in range(len(unique_vals))}
        
    df[col] = df[col].apply(lambda x: transform(x, vocab))
    return df[col], vocab

def fit_vocabulary(df_ratings:pd.DataFrame, df_movies:pd.DataFrame):
    vocabulary = {}
    max_vocab_size = {'userId':1e100, 'movieId':1e100, 'description':1e5, 'genres':100, 'movie_year':1e100}

    for col in ['userId', 'movieId']:
        print(col)
        df_ratings[col], v = categorical_encoding(df_ratings, col, max_vocab_size[col])
        vocabulary[col] = v

    for col in ['description', 'genres', 'movie_year']:
        print(col)
        df_movies[col], v = categorical_encoding(df_movies, col, max_vocab_size[col])
        vocabulary[col] = v
    
    return vocabulary, df_ratings, df_movies

def score_vocabulary(df_ratings:pd.DataFrame, vocabulary:dict):
    df_ratings = df_ratings.reset_index()
    for col in ['userId', 'movieId']:
        print(col)
        df_ratings[col] = df_ratings[col].apply(lambda x: transform(x, vocabulary[col]))
    
    return df_ratings


def get_historical_user_features_cpp(df:pd.DataFrame, max_hist=20):
    user_ids = df['userId'].to_numpy().astype(np.uint32)
    movie_ids = df['movieId'].to_numpy().astype(np.uint32)
    ratings = df['normalized_rating'].to_numpy().astype(np.float32)
    timestamps = df['timestamp'].to_numpy().astype(np.uint64)

    prev_movie_ids, prev_ratings  = ml_32m_py.py_get_historical_features(user_ids, movie_ids, timestamps, ratings, df.shape[0], max_hist)

    df["prev_movie_ids"] = prev_movie_ids
    df["prev_ratings"] = prev_ratings


def save_dfs_parquet(out_dir:str, vocabulary:dict, df_ratings_train:pd.DataFrame, df_ratings_val:pd.DataFrame, df_ratings_test:pd.DataFrame, df_movies:pd.DataFrame, num_partitions:int=32):
    df_ratings_train["partition"] = [random.randint(1, num_partitions) for _ in range(len(df_ratings_train))]
    df_ratings_val["partition"]   = [random.randint(1, num_partitions) for _ in range(len(df_ratings_val))]
    df_ratings_test["partition"]  = [random.randint(1, num_partitions) for _ in range(len(df_ratings_test))]

    if os.path.exists(out_dir):
        try:
            shutil.rmtree(out_dir)
        except:
            pass

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(vocabulary, f"{out_dir}/vocabulary.pkl")
    df_ratings_train.to_parquet(out_dir + "/train/", partition_cols=["partition"])
    df_ratings_val.to_parquet(out_dir + "/validation/", partition_cols=["partition"])
    df_ratings_test.to_parquet(out_dir + "/test/", partition_cols=["partition"])
    df_movies.to_parquet(out_dir + "/movies.parquet")


def upload_directory_with_transfer_manager(bucket_name:str, source_path:str, destination_path:str, workers=8):
    try:
        storage_client = Client()
        bucket = storage_client.bucket(bucket_name)

        directory_as_path_obj = Path(source_path)
        paths = directory_as_path_obj.rglob("*.parquet")

        file_paths = [path for path in paths if path.is_file()]
        relative_paths = [path.relative_to(source_path) for path in file_paths]

        string_paths = [str(path) for path in relative_paths]

        print("Found {} files.".format(len(string_paths)))

        if destination_path.endswith("/") is False:
            destination_path += "/"

        results = transfer_manager.upload_many_from_filenames(
            bucket, 
            string_paths, 
            blob_name_prefix=destination_path,
            source_directory=source_path, 
            max_workers=workers
        )

        for name, result in zip(string_paths, results):
            if isinstance(result, Exception):
                print("Failed to upload {} due to exception: {}".format(name, result))
            else:
                print("Uploaded {} to {}.".format(name, bucket.name))
    
    except Exception as e:
        print(e)


def delete_gcp_folder(bucket_name:str, folder_path:str):
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        prefix = folder_path if folder_path.endswith('/') else f"{folder_path}/"
        blobs = list(bucket.list_blobs(prefix=prefix))
        
        if blobs:
            bucket.delete_blobs(blobs)
            print(f"Deleted {len(blobs)} objects from {folder_path}")
        else:
            print("No objects found to delete.")

    except Exception as e:
        print(e)


def run_dp_pipeline(dataset_path):
    # print("Reading datasets from path...")
    # df_ratings, df_movies = get_ml_32m_dataframe(dataset_path)

    # print("Normalizing ratings...")
    # df_ratings = normalize_ratings(df_ratings)

    # print("Splitting into train test and validation...")
    # df_ratings_train, df_ratings_val, df_ratings_test = split_train_test(df_ratings, min_rated=10)

    # print("Fitting vocabulary...")
    # vocabulary, df_ratings_train, df_movies = fit_vocabulary(df_ratings_train, df_movies)

    # print("Vocabulary on validation...")
    # df_ratings_val = score_vocabulary(df_ratings_val, vocabulary)
    # print("Vocabulary on test...")
    # df_ratings_test = score_vocabulary(df_ratings_test, vocabulary)

    # print("Prepare historical features train...")
    # get_historical_user_features_cpp(df_ratings_train)
    # print("Prepare historical features val...")
    # get_historical_user_features_cpp(df_ratings_val)
    # print("Prepare historical features test...")
    # get_historical_user_features_cpp(df_ratings_test)

    # print("Saving parquet files...")
    # save_dfs_parquet("parquet_dataset_ml_32m", vocabulary, df_ratings_train, df_ratings_val, df_ratings_test, df_movies, num_partitions=32)

    print("Deleting existing folder in cloud...")
    delete_gcp_folder("r6-ae-dev-adperf-adintelligence-data", "amondal/parquet_dataset_ml_32m")

    print("Uploading to cloud...")
    upload_directory_with_transfer_manager("r6-ae-dev-adperf-adintelligence-data", "parquet_dataset_ml_32m", "amondal/parquet_dataset_ml_32m/")

if __name__ == '__main__':
    dataset_path = "datasets/ml-32m"
    run_dp_pipeline(dataset_path)


