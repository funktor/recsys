import argparse
import os
import torch
import pyarrow.parquet as pq
import gcsfs
import model
import dataloader
import torch.optim as optim

import math
import os
import numpy as np
import pandas as pd
import random
import uuid
import joblib

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.distributed as dist
from torch.distributed import init_process_group
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
from model import RecommenderSystem
import datasets
from datasets import Dataset
import gc

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["RANK"]))

def checkpoint(model:nn.Module, optimizer:torch.optim.Optimizer, filename):
    torch.save({'optimizer':optimizer.state_dict(), 'model':model.state_dict()}, filename)
    
def load_model(filename):
    chkpt = torch.load(filename, weights_only=False)
    return chkpt['model'], chkpt['optimizer']

def count_rows_in_gcs_parquet(parquet_path:str):
    """
    Counts the total number of rows across multiple Parquet files in a GCS bucket path.

    Args:
        bucket_path (str): The Google Cloud Storage path (e.g., "gs://your-bucket/your-folder/").

    Returns:
        int: The total number of rows.
    """
    # Initialize the GCSFileSystem
    fs = gcsfs.GCSFileSystem()
    
    # Use pyarrow to open the dataset without reading the actual data
    # parquet_path is assumed to be in the following format: gs://[bucket-name]/**/*.parquet
    parquet_paths = parquet_path.split("/")
    parquet_paths = parquet_paths[2:-1]
    parquet_dir = "/".join(parquet_paths)
    print(parquet_dir)

    dataset = pq.ParquetDataset(parquet_dir, filesystem=fs)
    
    # Sum the row counts from the metadata of each fragment (file)
    total_rows = sum(fragment.count_rows() for fragment in dataset.fragments)
    return total_rows


def get_trainer_and_optimizer(vocabulary:dict, rank:int):
    user_id_size = len(vocabulary["userId"])+1
    movie_id_size = len(vocabulary["movieId"])+1
    user_embedding_size = 128
    user_prev_rated_seq_len = 20
    user_num_encoder_layers = 1
    user_num_heads = 4
    user_dim_ff = 128
    user_dropout = 0.0
    movie_desc_size = len(vocabulary["description"])+1
    movie_genres_size = len(vocabulary["genres"])+1
    movie_year_size = len(vocabulary["movie_year"])+1
    movie_embedding_size = 128
    movie_dropout = 0.0
    embedding_size = 128

    rec = \
        RecommenderSystem(
            user_id_size, 
            user_embedding_size, 
            user_prev_rated_seq_len, 
            user_num_encoder_layers, 
            user_num_heads, 
            user_dim_ff,
            user_dropout,
            movie_id_size, 
            movie_desc_size,
            movie_genres_size,
            movie_year_size, 
            movie_embedding_size, 
            movie_dropout,
            embedding_size
        ).to(rank)
    
    optimizer = optim.Adam(rec.parameters(), lr=0.0001)
    return rec, optimizer


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
    

def save_movie_embeddings(model:RecommenderSystem, movies_dataset:Dataset, path:str, batch_size:int=1024, movie_emb_size:int=128):
    model.eval()
    with torch.no_grad():
        movie_emb_mmap = np.memmap(path, dtype=np.float32, mode="w+", shape=(movies_dataset.shape[0], movie_emb_size+1))
        movie_batch_iter = dataloader.get_unique_movies(movies_dataset, batch_size, device=0)
        
        i = 0
        while True:
            try:
                batch = next(movie_batch_iter)
                movie_ids, movie_descriptions, movie_genres, movie_years = batch

                output:torch.Tensor = \
                    model.get_movie_embeddings(
                        movie_ids, 
                        movie_descriptions, 
                        movie_genres, 
                        movie_years
                    )
                
                movie_emb_mmap[i:i+output.shape[0], 0] = movie_ids.cpu().numpy()
                movie_emb_mmap[i:i+output.shape[0], 1:] = output.cpu().numpy()
                i += output.shape[0]

            except StopIteration:
                break


def save_users_embeddings(model:RecommenderSystem, ratings_dataset:pd.DataFrame, path:str, batch_size:int=1024, users_emb_size:int=128):
    model.eval()
    with torch.no_grad():
        n = ratings_dataset['userId'].nunique()
        users_emb_mmap = np.memmap(path, dtype=np.float32, mode="w+", shape=(n, users_emb_size+1))
        users_batch_iter = dataloader.get_unique_users(ratings_dataset, batch_size, device=0)
        
        i = 0
        while True:
            try:
                user_ids = next(users_batch_iter)

                output:torch.Tensor = \
                    model.get_user_embeddings(
                        user_ids
                    )
                
                users_emb_mmap[i:i+output.shape[0], 0] = user_ids.cpu().numpy()
                users_emb_mmap[i:i+output.shape[0], 1:] = output.cpu().numpy()
                i += output.shape[0]

            except StopIteration:
                break
    
    
def train_func(config: dict):
    try:
        print("Setting up DDP...")
        if dist.is_initialized() is False:
            ddp_setup()

        rank_local  = int(os.environ["LOCAL_RANK"])
        rank_global = int(os.environ["RANK"])
        world_size  = int(os.environ["WORLD_SIZE"])

        print(f"WORLD SIZE = {world_size}")

        datasets_gcs_path = config["gcs_dir"]
        ratings_train_path = f"{datasets_gcs_path}/train/*.parquet"
        path_vocab = f"{datasets_gcs_path}/vocabulary.pkl"
        batch_size = config["batch_size"]
        max_num_epochs = config["num_epochs"]
        accumulate_grad_batches = config["accumulate_grad_batches"]
        model_path = config["existing_model_path"]
        max_num_batches = config["max_num_batches"]

        print("Getting datasets...")
        ratings_train, ratings_val, movies_dataset = dataloader.get_datasets(datasets_gcs_path, world_size, rank_global)

        num_train_data = count_rows_in_gcs_parquet(ratings_train_path)
        print(f"Total Training Data = {num_train_data}")
        print(f"Effective number of batches = {world_size*batch_size}")

        batches_per_epoch = num_train_data // (world_size*batch_size)
        batches_per_epoch = min(batches_per_epoch, max_num_batches)

        print(f"Rank={rank_global}: Train data size   = {ratings_train.shape[0]}")
        print(f"Rank={rank_global}: Val data size     = {ratings_val.shape[0]}")
        print(f"Rank={rank_global}: Batches per epoch = {batches_per_epoch}")

        if rank_local == 0:
            dataloader.download_vocabulary(path_vocab, "/tmp/vocabulary.pkl")
            Path('/tmp/marker_file.txt').touch()
        else:
            while True:
                if os.path.exists('/tmp/marker_file.txt'):
                    break

        print("Reading vocabulary...")
        vocabulary = dataloader.get_vocabulary("/tmp/vocabulary.pkl")

        print("Getting model and optimizer...")
        rec, optimizer = get_trainer_and_optimizer(vocabulary, rank_global)

        if model_path and os.path.exists(model_path):
            rec_st, optimizer_st = load_model(model_path)
            rec.load_state_dict(rec_st)
            optimizer.load_state_dict(optimizer_st)
        
        rec = DDP(rec, device_ids=[rank_global], find_unused_parameters=True)
        best_vloss = float("Inf")

        if rank_local == 0:
            model_out_dir = "/tmp/model_outputs"
            os.makedirs(model_out_dir, exist_ok=True)

        optimizer.zero_grad()

        criterion = nn.MSELoss()
        scheduler = CosineWarmupScheduler(optimizer, warmup=50, max_iters=batches_per_epoch*max_num_epochs/accumulate_grad_batches)

        for epoch in range(max_num_epochs):
            print(f"Starting epoch {epoch+1}...")
            rec.train()

            sum_loss = 0.0
            sum_rows = 0

            batch_iter = dataloader.prepare_batches_prefetch(ratings_train, movies_dataset, batch_size, device=rank_global)
            i = 0
            while True:
                try:
                    batch = next(batch_iter)

                    data, labels = batch
                    user_ids, user_prev_rated_movie_ids, user_prev_ratings, movie_ids, movie_descriptions, movie_genres, movie_years = data

                    output:torch.Tensor = \
                        rec(
                            user_ids,
                            user_prev_rated_movie_ids, 
                            user_prev_ratings,
                            movie_ids, 
                            movie_descriptions, 
                            movie_genres, 
                            movie_years
                        )
                    
                    batch_loss:torch.Tensor = criterion(output.contiguous(), labels.contiguous())

                    sum_loss += output.shape[0]*batch_loss.item()
                    sum_rows += output.shape[0]

                    batch_loss /= accumulate_grad_batches
                    batch_loss.backward()

                    if (i+1) % accumulate_grad_batches == 0:
                        acc_loss = torch.Tensor([sum_loss, sum_rows]).to(rank_global)
                        dist.reduce(acc_loss, dst=0, op=dist.ReduceOp.SUM)

                        if rank_local == 0:
                            acc_loss = acc_loss.tolist()
                            avg_loss = acc_loss[0]/acc_loss[1]
                            print(f"Epoch: {epoch+1}, Batch: {i+1}, Average Loss: {avg_loss}")

                        nn.utils.clip_grad_norm_(rec.parameters(), max_norm=1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                except StopIteration:
                    break

                i += 1

                if i >= batches_per_epoch:
                    break


            acc_loss = torch.Tensor([sum_loss, sum_rows]).to(rank_global)
            dist.reduce(acc_loss, dst=0, op=dist.ReduceOp.SUM)

            if rank_local == 0:
                acc_loss = acc_loss.tolist()
                avg_loss = acc_loss[0]/acc_loss[1]
                print(f"Epoch: {epoch+1}, Batch: {i+1}, Average Loss: {avg_loss}")

            nn.utils.clip_grad_norm_(rec.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()


            print(f"Running validation for epoch {epoch+1}...")
            rec.eval()

            with torch.no_grad():
                batch_iter_val = dataloader.prepare_batches_prefetch(ratings_val, movies_dataset, 32, device=rank_global)
                sum_loss = 0.0
                sum_rows = 0

                i = 0
                while True:
                    try:
                        batch = next(batch_iter_val)

                        data, labels = batch
                        user_ids, user_prev_rated_movie_ids, user_prev_ratings, movie_ids, movie_descriptions, movie_genres, movie_years = data

                        output:torch.Tensor = \
                            rec(
                                user_ids,
                                user_prev_rated_movie_ids, 
                                user_prev_ratings,
                                movie_ids, 
                                movie_descriptions, 
                                movie_genres, 
                                movie_years
                            )
                    
                        batch_loss:torch.Tensor = criterion(output.contiguous(), labels.contiguous())

                        sum_loss += output.shape[0]*batch_loss.item()
                        sum_rows += output.shape[0]

                        if rank_local == 0:
                            print(f"Epoch: {epoch+1}, Batch: {i+1}, Loss (Rank 0): {sum_loss/sum_rows}")

                    except StopIteration:
                        break

                    i += 1

                    if i >= max_num_batches:
                        break
                
                vloss = torch.Tensor([sum_loss, sum_rows]).to(rank_global)
                dist.reduce(vloss, dst=0, op=dist.ReduceOp.SUM)

                if rank_local == 0:
                    vloss = vloss.tolist()
                    avg_vloss = vloss[0]/vloss[1]
                    print(f"Average Validation Loss: {avg_vloss}")
                    print()
                    print("Checkpointing...")
                    
                    if avg_vloss < best_vloss:
                        best_vloss = avg_vloss
                        checkpoint(rec.module, optimizer, os.path.join(model_out_dir, f"checkpoint-best-vloss.pth"))
        
        model:RecommenderSystem = rec.module

        if rank_local == 0:
            print("Saving model...")
            checkpoint(model, optimizer, os.path.join(model_out_dir, f"final_model.pth"))
        
        if rank_local == 0:
            print("Saving movie embeddings...")
            os.makedirs("movie_embeddings", exist_ok=True)
            save_movie_embeddings(
                model, 
                movies_dataset, 
                "movie_embeddings/embeds.mmap", 
                batch_size=1024, 
                movie_emb_size=model.movie_embedding_size
            )
            
            print("Getting all ratings train data...")
            all_files = dataloader.pre_partitions_for_download(f"{datasets_gcs_path}/train", 1, 0)
            cache_dir_full_train = f"/tmp/huggingface/{rank_local}/full_train"
            os.makedirs(cache_dir_full_train, exist_ok=True)
            ratings_train_full = datasets.load_dataset("parquet", data_files=all_files, split="train", cache_dir=cache_dir_full_train)
            ratings_train_full = ratings_train_full.select_columns("userId")
            ratings_train_full = ratings_train_full.to_pandas()

            print("Saving users embeddings...")
            os.makedirs("users_embeddings", exist_ok=True)
            save_users_embeddings(
                model, 
                ratings_train_full, 
                "users_embeddings/embeds.mmap", 
                batch_size=1024, 
                users_emb_size=model.user_embedding_size
            )
            
    except Exception as e:
        print(e)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ray-based UEM Model Training')

    parser.add_argument('--gcs_dir', type=str, required=True,
                        help='Path to GCS directory')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size per worker')
    parser.add_argument('--max_num_batches', type=int, default=1e9,
                        help='Maximum batches for debugging')
    parser.add_argument('--num_epochs', type=int, default=40,
                        help='Maximum number of training epochs')
    parser.add_argument('--accumulate_grad_batches', type=int, default=4,
                        help='Accumulate gradients over N batches')
    parser.add_argument('--existing_model_path', type=str, default=None,
                        help='Use existing model')

    args = parser.parse_args()

    train_func(vars(args))
    # torchrun --standalone --nproc_per_node=8 trainer.py --gcs_dir "gs://r6-ae-dev-adperf-adintelligence-data/amondal/parquet_dataset_ml_32m" --batch_size 128 --num_epochs 10 --accumulate_grad_batches 4
