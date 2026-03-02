import os
import datasets
import pandas as pd
import numpy as np
import torch
from typing import List
import fsspec
from google.cloud import storage
import joblib
from datasets import Dataset
import multiprocessing
from multiprocessing import Queue

def pre_partitions_with_files(filepaths:List[str], world_size, rank):
    """
    Partition cloud folder files equally to each GPU worker
    """
    train_files = []
    for i in range(len(filepaths)):
        if i % world_size == rank:
            train_files += [f"gs://{filepaths[i]}"]

    print(f"Files assigned to GPU with {rank}: {train_files[:min(10, len(train_files))]}...")

    return train_files


def pre_partitions_for_download(path:str, world_size, rank):
    """
    Partition cloud folder files equally to each GPU worker
    """
    fs = fsspec.filesystem("gcs")
    partitions = fs.glob(f"{path}/**/*.parquet")
    partitions = sorted(partitions)

    if not partitions:
        raise ValueError(f"No partitions found matching pattern: {path}")

    train_files = pre_partitions_with_files(partitions, world_size, rank)
    return train_files

def download_vocabulary(gs_path:str, local_path:str):
    """
    Download vocabulary from GCS
    """
    storage_client = storage.Client()
    path_splits = gs_path.split('/')
    bucket_name = path_splits[2]
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob('/'.join(path_splits[3:]))
    blob.download_to_filename(local_path)

def get_vocabulary(local_path:str):
    vocabulary = joblib.load(local_path)
    return vocabulary

def get_dataset(path:str, cache_dir:str, world_size:int, rank:int, in_memory:bool=False, path_is_dir:bool=True):
    """
    Get dataset by prepartitioning files to each worker process
    """
    if path_is_dir:
        files = pre_partitions_for_download(path, world_size, rank)
    else:
        files = path

    os.makedirs(cache_dir, exist_ok=True)

    dataset = datasets.load_dataset("parquet", data_files=files, split="train", cache_dir=cache_dir, keep_in_memory=in_memory)
    return dataset


def get_datasets(path:str, world_size:int, rank:int):
    """
    Get train, validation and movies datasets
    """
    ratings_train = \
        get_dataset(
            f"{path}/train", 
            f"/tmp/huggingface/{rank}/train", 
            world_size, 
            rank
        ).set_format('pandas')
    
    ratings_val = \
        get_dataset(
            f"{path}/validation", 
            f"/tmp/huggingface/{rank}/val", 
            world_size, 
            rank
        ).set_format('pandas')
    
    movies_dataset = \
        get_dataset(
            f"{path}/movies.parquet", 
            f"/tmp/huggingface/{rank}/movies", 
            world_size, 
            rank, 
            in_memory=True, 
            path_is_dir=False
        ).to_pandas()

    return ratings_train, ratings_val, movies_dataset


def pad_batch(values, dtype, max_seq_len=None):
    """
    Pad batch
    """
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
    """
    Prepare batch by padding and converting to numpy and tensor formats
    """
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

        labels = df_ratings_batch_df["normalized_rating"].to_numpy(dtype=np.float32)

        del df_ratings_batch_df

        user_ids = torch.from_numpy(user_ids).pin_memory()
        user_prev_rated_movie_ids = torch.from_numpy(user_prev_rated_movie_ids).pin_memory()
        user_prev_ratings = torch.from_numpy(user_prev_ratings).pin_memory()

        movie_ids = torch.from_numpy(movie_ids).pin_memory()
        movie_descriptions = torch.from_numpy(movie_descriptions).pin_memory()
        movie_genres = torch.from_numpy(movie_genres).pin_memory()
        movie_years = torch.from_numpy(movie_years).pin_memory()

        labels = torch.from_numpy(labels).pin_memory()

        yield [user_ids, user_prev_rated_movie_ids, user_prev_ratings, movie_ids, movie_descriptions, movie_genres, movie_years], labels


def fill_prefetch_queue(queue:Queue, batch_iter, stream, device):
    """
    Method to fetch batch and push to queue
    """
    try:
        data, labels = next(batch_iter)
    except StopIteration:
        queue.put(None)
        return 0
    
    with torch.cuda.stream(stream):
        data_gpu = []
        for obj in data:
            data_gpu += [obj.to(device=device, non_blocking=True)]

        labels_gpu = labels.to(device=device, non_blocking=True)
        
        for obj in data_gpu:
            obj.record_stream(stream)
        
        labels_gpu.record_stream(stream)
        queue.put((data_gpu, labels_gpu))
    
    del data
    del labels
    return 1


def fill_queue(queue:Queue, max_num_items:int, batch_iter, stream, device):
    """
    Method called by each producer process
    """
    while len(queue) < max_num_items:
        res = fill_prefetch_queue(queue, batch_iter, stream, device)
        if res == 0:
            break


def prepare_batches_prefetch(ratings_dataset:Dataset, movies_dataset:pd.DataFrame, batch_size=128, device="gpu", prefetch_factor:int=16, num_workers:int=4):
    """
    Get batches using prefetching through multiple workers
    """
    stream = torch.cuda.Stream()
    batch_iter = prepare_batches(ratings_dataset, movies_dataset, batch_size, device)

    # multiprocessing queue to push the prefetched batches
    queue = Queue()

    # Each producer process gets batches and pushes to queue
    producers = []
    for _ in range(num_workers):
        p = multiprocessing.Process(target=fill_queue, args=(queue, prefetch_factor*num_workers, batch_iter, stream, device))
        p.start()
        producers += [p]

    # Main consumer process from queue
    while True:
        batch = queue.get()
        if batch is None:
            break
        data, labels = batch
        torch.cuda.current_stream().wait_stream(stream)
        yield data, labels

    for p in producers:
        p.join()


def get_unique_movies(movies_dataset:pd.DataFrame, batch_size=128, device="gpu"):
    """
    Get unique movies
    """
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
    """
    Get unique users from ratings file
    """
    ratings_dataset_df = ratings_dataset.drop_duplicates()
    n = ratings_dataset_df.shape[0]

    for i in range(0, n, batch_size):
        df_ratings_batch_df:pd.DataFrame = ratings_dataset_df[i:min(n,i+batch_size)]
        df_ratings_batch_df = df_ratings_batch_df.reset_index()

        user_ids = df_ratings_batch_df["userId"].to_numpy(dtype=np.int32)
        user_ids = torch.from_numpy(user_ids).pin_memory(device=device).to(device=device, non_blocking=True)

        yield user_ids




