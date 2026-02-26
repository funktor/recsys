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
from collections import deque

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

def download_vocabulary(gs_path:str, local_path:str):
    storage_client = storage.Client()
    path_splits = gs_path.split('/')
    bucket_name = path_splits[2]
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob('/'.join(path_splits[3:]))
    blob.download_to_filename(local_path)

def get_vocabulary(local_path:str):
    vocabulary = joblib.load(local_path)
    return vocabulary


def get_datasets(path:str, world_size:int, rank_local:int):
    train_files = pre_partitions_for_download(f"{path}/train", world_size, rank_local)
    cache_dir_train = f"/tmp/huggingface/{rank_local}/train"
    os.makedirs(cache_dir_train, exist_ok=True)
    ratings_train = datasets.load_dataset("parquet", data_files=train_files, split="train", cache_dir=cache_dir_train)
    ratings_train.set_format("pandas")

    val_files = pre_partitions_for_download(f"{path}/validation", world_size, rank_local)
    cache_dir_val = f"/tmp/huggingface/{rank_local}/val"
    os.makedirs(cache_dir_val, exist_ok=True)
    ratings_val = datasets.load_dataset("parquet", data_files=val_files, split="train", cache_dir=cache_dir_val)
    ratings_val.set_format("pandas")

    cache_dir_movies = f"/tmp/huggingface/{rank_local}/movies"
    os.makedirs(cache_dir_movies, exist_ok=True)
    movies_dataset = datasets.load_dataset("parquet", data_files=f"{path}/movies.parquet", split="train", keep_in_memory=True, cache_dir=cache_dir_movies)
    movies_dataset = movies_dataset.to_pandas()

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


def prepare_batches(ratings_dataset:Dataset, movies_dataset:pd.DataFrame, batch_size=128, device="gpu"):
    max_seq_len = 20
    n = ratings_dataset.shape[0]

    for i in range(0, n, batch_size):
        df_ratings_batch_df:pd.DataFrame = ratings_dataset[i:min(n,i+batch_size)]
        df_ratings_batch_df = df_ratings_batch_df.reset_index()
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

        user_ids = torch.from_numpy(user_ids).pin_memory(device='cpu')
        user_prev_rated_movie_ids = torch.from_numpy(user_prev_rated_movie_ids).pin_memory(device='cpu')
        user_prev_ratings = torch.from_numpy(user_prev_ratings).pin_memory(device='cpu')

        movie_ids = torch.from_numpy(movie_ids).pin_memory(device='cpu')
        movie_descriptions = torch.from_numpy(movie_descriptions).pin_memory(device='cpu')
        movie_genres = torch.from_numpy(movie_genres).pin_memory(device='cpu')
        movie_years = torch.from_numpy(movie_years).pin_memory(device='cpu')

        labels = torch.from_numpy(df_ratings_batch_df["normalized_rating"].to_numpy(dtype=np.float32)).pin_memory(device='cpu')

        yield [user_ids, user_prev_rated_movie_ids, user_prev_ratings, movie_ids, movie_descriptions, movie_genres, movie_years], labels


def fill_prefetch_queue(queue:deque, batch_iter, stream, device):
    try:
        data, labels = next(batch_iter)
    except StopIteration:
        queue.append(None)
        return
    
    with torch.cuda.stream(stream):
        data_gpu = []
        for obj in data:
            if isinstance(obj, torch.Tensor):
                data_gpu += [obj.to(device=device, non_blocking=True)]
                obj.record_stream(stream)

        if isinstance(labels, torch.Tensor):
            labels_gpu = labels.to(device=device, non_blocking=True)
            labels.record_stream(stream)

        queue.append((data_gpu, labels_gpu))

def prepare_batches_prefetch(ratings_dataset:Dataset, movies_dataset:pd.DataFrame, batch_size=128, device="gpu", prefetch_factor:int=4):
    stream = torch.cuda.Stream()
    batch_iter = prepare_batches(ratings_dataset, movies_dataset, batch_size, device)
    queue = deque()

    for _ in range(prefetch_factor):
        fill_prefetch_queue(queue, batch_iter, stream, device)
    
    while True:
        batch = queue.popleft()
        if batch is not None:
            data, labels = batch
            torch.cuda.current_stream().wait_stream(stream)
            fill_prefetch_queue(queue, batch_iter, stream, device)
            yield data, labels
        else:
            break


def get_unique_movies(movies_dataset:pd.DataFrame, batch_size=128, device="gpu"):
    n = movies_dataset.shape[0]

    for i in range(0, n, batch_size):
        movies_dataset_batch_df:pd.DataFrame = movies_dataset[i:min(n,i+batch_size)]
        movies_dataset_batch_df = movies_dataset_batch_df.reset_index()

        movie_ids = movies_dataset_batch_df["movieId"].to_numpy(dtype=np.int32)
        movie_descriptions = pad_batch(movies_dataset_batch_df["description"].to_numpy(), dtype=np.int32)
        movie_genres = pad_batch(movies_dataset_batch_df["genres"].to_numpy(), dtype=np.int32)
        movie_years = movies_dataset_batch_df["movie_year"].to_numpy(dtype=np.int32)

        movie_ids = torch.from_numpy(movie_ids).pin_memory(device=device).to(device=device, non_blocking=True)
        movie_descriptions = torch.from_numpy(movie_descriptions).pin_memory(device=device).to(device=device, non_blocking=True)
        movie_genres = torch.from_numpy(movie_genres).pin_memory(device=device).to(device=device, non_blocking=True)
        movie_years = torch.from_numpy(movie_years).pin_memory(device=device).to(device=device, non_blocking=True)

        yield [movie_ids, movie_descriptions, movie_genres, movie_years]


def get_unique_users(ratings_dataset:pd.DataFrame, batch_size=128, device="gpu"):
    ratings_dataset_df = ratings_dataset.drop_duplicates()
    n = ratings_dataset_df.shape[0]

    for i in range(0, n, batch_size):
        df_ratings_batch_df:pd.DataFrame = ratings_dataset_df[i:min(n,i+batch_size)]
        df_ratings_batch_df = df_ratings_batch_df.reset_index()

        user_ids = df_ratings_batch_df["userId"].to_numpy(dtype=np.int32)
        user_ids = torch.from_numpy(user_ids).pin_memory(device=device).to(device=device, non_blocking=True)

        yield user_ids




