import traceback
import os

import datasets
import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from typing import Optional, List
from torch.utils.data.distributed import DistributedSampler
import fsspec
from google.cloud import storage
import joblib
from datasets import Dataset

# try:
#     from ray.train import get_context as ray_get_ctx
# except Exception:
#     ray_get_ctx = None


def get_world_info():
    """
    Get world size and rank strictly from Ray Train.
    Raises RuntimeError if not running inside a Ray Train session.
    """
    try:
        import ray.train
        # ray.train.get_context() raises RuntimeError if session is missing
        ctx = ray.train.get_context()
        return ctx.get_world_size(), ctx.get_world_rank()

    except (ImportError, RuntimeError) as e:
        # Catch ImportError (Ray not installed) or RuntimeError (No session)
        raise RuntimeError(
            "Distributed context missing: This script must be launched via Ray Train."
        ) from e

def pre_partitions_with_files(filepaths:List[str], world_size, rank):
    train_files = []
    for i in range(len(filepaths)):
        if i % world_size == rank:
            train_files += [f"gs://{filepaths[i]}"]

    print(f"Files assigned to GPU with {rank}: {train_files[:min(10, len(train_files))]}...")

    return train_files


def pre_partitions_for_download(path:str, world_size, rank):
    fs = fsspec.filesystem("gcs")
    partitions = fs.glob(f"{path}/**/*.parquet")
    partitions = sorted(partitions)

    if not partitions:
        raise ValueError(f"No partitions found matching pattern: {path}")

    train_files = pre_partitions_with_files(partitions, world_size, rank)
    return train_files


def get_vocabulary(gs_path:str, local_path:str):
    storage_client = storage.Client()
    path_splits = gs_path.split('/')
    bucket_name = path_splits[2]
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob('/'.join(path_splits[3:]))
    blob.download_to_filename(local_path)
    vocabulary = joblib.load(local_path)
    return vocabulary


def get_datasets(path:str):
    # world_size, rank = get_world_info()
    # print(f"WORLD_SIZE={world_size} RANK={rank}")

    world_size, rank = 1, 0
    
    train_files = pre_partitions_for_download(f"{path}/train", world_size, rank)
    ratings_train = datasets.load_dataset("parquet", data_files=train_files, split="train", cache_dir="/tmp/huggingface")
    ratings_train.set_format("pandas")

    val_files = pre_partitions_for_download(f"{path}/validation", world_size, rank)
    ratings_val = datasets.load_dataset("parquet", data_files=val_files, split="train")
    ratings_val.set_format("pandas")

    movies_dataset = datasets.load_dataset("parquet", data_files=f"{path}/movies.parquet", split="train", keep_in_memory=True)
    movies_dataset.set_format("pandas")

    return ratings_train, ratings_val, movies_dataset


def pad_batch(values, dtype, max_seq_len=None):
    if max_seq_len is None:
        max_seq_len = max([len(x) for x in values])
    
    arr = np.zeros((len(values), max_seq_len), dtype=dtype)

    for i in range(len(values)):
        k = max_seq_len-1
        for j in range(len(values[i])-1, -1, -1):
            arr[i,k] = values[i][j]
            k -= 1
    
    return arr


def prepare_batches(ratings_dataset:Dataset, movies_dataset:Dataset, batch_size=128, device="gpu", batch_limit=None):
    max_seq_len = 20
    n = ratings_dataset.shape[0]

    k = 0
    for i in range(0, n, batch_size):
        df_ratings_batch_df:pd.DataFrame = ratings_dataset[i:min(n,i+batch_size)]
        df_ratings_batch_df = df_ratings_batch_df.merge(movies_dataset, on=["movieId"], how="left")

        df_ratings_batch_df['description'] = df_ratings_batch_df['description'].apply(lambda x: x if isinstance(x, list) else [])
        df_ratings_batch_df['genres'] = df_ratings_batch_df['genres'].apply(lambda x: x if isinstance(x, list) else [])
        df_ratings_batch_df['movie_year'] = df_ratings_batch_df['movie_year'].fillna(0)

        user_ids = df_ratings_batch_df["userId"].to_numpy(dtype=np.int32)
        user_prev_rated_movie_ids = pad_batch(df_ratings_batch_df["prev_movie_ids"].to_numpy(), dtype=np.int32, max_seq_len=max_seq_len)
        user_prev_ratings = pad_batch(df_ratings_batch_df["prev_ratings"].to_numpy(), dtype=np.float32, max_seq_len=max_seq_len)

        movie_ids = df_ratings_batch_df["movieId"].to_numpy(dtype=np.int32)
        movie_descriptions = pad_batch(df_ratings_batch_df["description"].to_numpy(), dtype=np.int32)
        movie_genres = pad_batch(df_ratings_batch_df["genres"].to_numpy(), dtype=np.int32)
        movie_years = df_ratings_batch_df["movie_year"].to_numpy(dtype=np.int32)

        user_ids = torch.from_numpy(user_ids).to(device=device)
        user_prev_rated_movie_ids = torch.from_numpy(user_prev_rated_movie_ids).to(device=device)
        user_prev_ratings = torch.from_numpy(user_prev_ratings).to(device=device)

        movie_ids = torch.from_numpy(movie_ids).to(device=device)
        movie_descriptions = torch.from_numpy(movie_descriptions).to(device=device)
        movie_genres = torch.from_numpy(movie_genres).to(device=device)
        movie_years = torch.from_numpy(movie_years).to(device=device)

        labels = torch.from_numpy(df_ratings_batch_df["normalized_rating"].to_numpy(dtype=np.float32)).to(device=device)

        yield [user_ids, user_prev_rated_movie_ids, user_prev_ratings, movie_ids, movie_descriptions, movie_genres, movie_years], labels
        
        k += 1
        if batch_limit is not None and k >= batch_limit:
            break


