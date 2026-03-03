import argparse
import os
import torch
import pyarrow.parquet as pq
import gcsfs
import dataloader
import torch.optim as optim

import os
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.distributed import init_process_group
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
from model import RecommenderSystem
from datasets import Dataset
import gc
import torch.multiprocessing as mp
import time
from data_generator import upload_directory_with_transfer_manager
import joblib

mp.set_start_method('spawn', force=True)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def ddp_setup():
    """
    Setup DDP
    """
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def checkpoint(model:nn.Module, optimizer:torch.optim.Optimizer, filename):
    """
    Checkpoint model and optimizer
    """
    torch.save({'optimizer':optimizer.state_dict(), 'model':model.state_dict()}, filename)
    
def load_model(filename):
    """
    Load model and optimizer
    """
    chkpt = torch.load(filename, weights_only=False)
    return chkpt['model'], chkpt['optimizer']

def count_rows_in_gcs_parquet(parquet_path:str):
    """
    Counts the total number of rows across multiple Parquet files in a GCS bucket path.
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
    """
    Get model and optimizer
    """
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
    """
    Cosine Learning Rate Scheduler
    """
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
    

def save_movie_embeddings(
        model:RecommenderSystem, 
        movies_dataset:Dataset, 
        path:str, 
        batch_size:int=1024, 
        movie_emb_size:int=128
    ):
    """
    Save movie embeddings as numpy memory mapped files
    """
    model.eval()
    with torch.no_grad():
        # get movie ids from movies dataset
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
                
                # save as numpy memmap
                movie_emb_mmap[i:i+output.shape[0], 0] = movie_ids.cpu().numpy()
                movie_emb_mmap[i:i+output.shape[0], 1:] = output.cpu().numpy()
                movie_emb_mmap.flush()
                i += output.shape[0]

            except StopIteration:
                break
        
        return (movies_dataset.shape[0], movie_emb_size+1)


def save_users_embeddings(
        model:RecommenderSystem, 
        ratings_dataset:pd.DataFrame, 
        path:str, 
        batch_size:int=1024, 
        users_emb_size:int=128,
        device:int=0
    ):
    """
    Save user embeddings as numpy memory mapped files
    """
    model.eval()
    with torch.no_grad():
        # get unique users from ratings dataset
        n = ratings_dataset['userId'].nunique()
        users_emb_mmap = np.memmap(path, dtype=np.float32, mode="w+", shape=(n, users_emb_size+1))
        users_batch_iter = dataloader.get_unique_users(ratings_dataset, batch_size, device=device)
        
        i = 0
        while True:
            try:
                user_ids = next(users_batch_iter)

                output:torch.Tensor = \
                    model.get_user_embeddings(
                        user_ids
                    )
                
                # save as numpy memmap
                users_emb_mmap[i:i+output.shape[0], 0] = user_ids.cpu().numpy()
                users_emb_mmap[i:i+output.shape[0], 1:] = output.cpu().numpy()
                users_emb_mmap.flush()
                i += output.shape[0]

            except StopIteration:
                break
        
        return (n, users_emb_size+1)

def save_embeddings_and_metadata(
        model:RecommenderSystem, 
        datasets_gcs_path:Dataset, 
        movies_dataset:pd.DataFrame, 
        gcs_bucket_name:str, 
        gcs_prefix:str, 
        world_size:int, 
        rank_global:int, 
        rank_local:int
    ):
    """
    Calculate embeddings and upload to GCS bucket
    """

    MOVIE_EMBEDDINGS_PATH = "movie_embeddings"
    USER_EMBEDDINGS_PATH  = "users_embeddings"

    # Save movie embeddings by rank=0 worker
    if rank_global == 0:
        print("Saving movie embeddings...")

        os.makedirs(MOVIE_EMBEDDINGS_PATH, exist_ok=True)
        movie_embeds_shape = \
            save_movie_embeddings(
                model, 
                movies_dataset, 
                f"{MOVIE_EMBEDDINGS_PATH}/embeds.mmap", 
                batch_size=1024, 
                movie_emb_size=model.movie_embedding_size
            )
        
        joblib.dump(movie_embeds_shape, f"{MOVIE_EMBEDDINGS_PATH}/movie_embeds_shape.pkl")

        print("Uploading movie embeddings to GCS...")
        upload_directory_with_transfer_manager(
            gcs_bucket_name, 
            MOVIE_EMBEDDINGS_PATH, 
            f"{gcs_prefix}/{MOVIE_EMBEDDINGS_PATH}"
        )
        
    # Save user embeddings by all workers
    print("Getting all ratings train data...")
    # Load full data
    ratings_train_full = \
        dataloader.get_dataset(
            f"{datasets_gcs_path}/full_data", 
            f"/tmp/huggingface/{rank_local}/full_data", 
            world_size=world_size, 
            rank=rank_global
        ).select_columns("userId").to_pandas()

    print("Saving users embeddings...")
    os.makedirs(f"{USER_EMBEDDINGS_PATH}/{rank_global}", exist_ok=True)
    user_embeds_shape = \
        save_users_embeddings(
            model, 
            ratings_train_full, 
            f"{USER_EMBEDDINGS_PATH}/{rank_global}/embeds.mmap", 
            batch_size=1024, 
            users_emb_size=model.user_embedding_size,
            device=rank_local
        )

    joblib.dump(user_embeds_shape, f"{USER_EMBEDDINGS_PATH}/{rank_global}/user_embeds_shape.pkl")

    print("Uploading user embeddings to GCS...")
    upload_directory_with_transfer_manager(
        gcs_bucket_name, 
        f"{USER_EMBEDDINGS_PATH}/{rank_global}", 
        f"{gcs_prefix}/{USER_EMBEDDINGS_PATH}/{rank_global}"
    )
    
    
def train_func(config: dict):
    """
    Main training method
    """
    # try:
    print("Setting up DDP...")
    if dist.is_initialized() is False:
        ddp_setup()

    VOCAB_PATH = "/tmp/vocabulary.pkl"
    VOCAB_MARKER_PATH = '/tmp/marker_file.txt'

    rank_local  = int(os.environ["LOCAL_RANK"])
    rank_global = int(os.environ["RANK"])
    world_size  = int(os.environ["WORLD_SIZE"])

    print(f"WORLD SIZE = {world_size}")

    gcs_bucket_name = config["gcs_bucket"]
    gcs_prefix = config["gcs_prefix"]
    gcs_data_dir = config["gcs_data_dir"]
    batch_size = config["batch_size"]
    max_num_epochs = config["num_epochs"]
    accumulate_grad_batches = config["accumulate_grad_batches"]
    model_path = config["existing_model_path"]
    max_num_batches = config["max_num_batches"]
    num_workers = min(mp.cpu_count()-1, config["num_workers"])
    model_out_dir = config["model_out_dir"]

    datasets_gcs_path = f"gs://{gcs_bucket_name}/{gcs_prefix}/{gcs_data_dir}"

    ratings_train_path = f"{datasets_gcs_path}/train/*.parquet"
    path_vocab = f"{datasets_gcs_path}/vocabulary.pkl"

    # Get datasets as huggingface datasets format
    print("Getting datasets...")
    ratings_train, ratings_val, movies_dataset = dataloader.get_datasets(datasets_gcs_path, world_size, rank_global)

    # Assign each GPU equal number of batches
    num_train_data = count_rows_in_gcs_parquet(ratings_train_path)
    print(f"Total Training Data = {num_train_data}")
    print(f"Effective batch size = {world_size*batch_size}")

    batches_per_epoch = num_train_data // (world_size*batch_size)
    batches_per_epoch = min(batches_per_epoch, max_num_batches)

    print(f"Rank={rank_global}: Train data size   = {ratings_train.shape[0]}")
    print(f"Rank={rank_global}: Val data size     = {ratings_val.shape[0]}")
    print(f"Rank={rank_global}: Batches per epoch = {batches_per_epoch}")

    # Download vocabulary to local path only by rank=0 worker. Need to synchronize using marker file 
    if rank_local == 0:
        dataloader.download_vocabulary(path_vocab, VOCAB_PATH)
        Path(VOCAB_MARKER_PATH).touch()
    else:
        while True:
            if os.path.exists(VOCAB_MARKER_PATH):
                # delete the marker file after every run
                break

    print("Reading vocabulary...")
    vocabulary = dataloader.get_vocabulary(VOCAB_PATH)

    print("Getting model and optimizer...")
    rec, optimizer = get_trainer_and_optimizer(vocabulary, rank_local)

    # Load existing model if model_path provided
    if model_path and os.path.exists(model_path):
        rec_st, optimizer_st = load_model(model_path)
        rec.load_state_dict(rec_st)
        optimizer.load_state_dict(optimizer_st)
    
    # Wrap model in DDP
    rec = DDP(rec, device_ids=[rank_local], find_unused_parameters=True)
    best_vloss = float("Inf")

    # Create path for storing checkpoints and saving final model
    if rank_global == 0:
        os.makedirs(model_out_dir, exist_ok=True)

    # Initialize optimizer
    optimizer.zero_grad()

    # Initialize loss
    criterion = nn.MSELoss()
    scheduler = \
        CosineWarmupScheduler(
            optimizer, 
            warmup=50, 
            max_iters=batches_per_epoch*max_num_epochs/accumulate_grad_batches
        )

    for epoch in range(max_num_epochs):
        print(f"Starting epoch {epoch+1}...")
        start_epoch_time = time.time()
        rec.train()

        sum_loss = 0.0
        sum_rows = 0

        # Get batch iterator
        batch_iter = dataloader.prepare_batches_prefetch(ratings_train, movies_dataset, rank_local, batch_size, device=rank_local, num_workers=num_workers)
        i = 0
        while True:
            try:
                # Get next batch of data and labels
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
                
                # Calculate batch loss
                batch_loss:torch.Tensor = criterion(output.contiguous(), labels.contiguous())
                batch_loss /= accumulate_grad_batches

                sum_loss += output.shape[0]*batch_loss.item()
                sum_rows += output.shape[0]
                
                batch_loss.backward()

                # Accumulate batches to compute gradient
                if (i+1) % accumulate_grad_batches == 0:
                    # Broadcast total loss and total number of rows to all gpu workers to calculate avg loss
                    acc_loss = torch.Tensor([sum_loss, sum_rows]).to(rank_local)
                    dist.reduce(acc_loss, dst=0, op=dist.ReduceOp.SUM)
                    acc_loss = acc_loss.tolist()
                    avg_loss = acc_loss[0]/acc_loss[1]

                    # Print loss by rank=0 worker
                    if rank_global == 0:
                        print(f"Epoch: {epoch+1}, Batch: {i+1}, Average Loss: {avg_loss}")

                    # Compute gradients
                    nn.utils.clip_grad_norm_(rec.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            except StopIteration:
                break

            i += 1

            if i >= batches_per_epoch:
                break

        print(f"{rank_local} is here2....")
        # Do same for remaining batches (not divisible by accumulate grad batches)
        acc_loss = torch.Tensor([sum_loss, sum_rows]).to(rank_local)
        dist.reduce(acc_loss, dst=0, op=dist.ReduceOp.SUM)
        acc_loss = acc_loss.tolist()
        avg_loss = acc_loss[0]/acc_loss[1]

        if rank_global == 0:
            print(f"Epoch: {epoch+1}, Batch: {i+1}, Average Loss: {avg_loss}")

        nn.utils.clip_grad_norm_(rec.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        end_epoch_time = time.time()

        duration = (end_epoch_time-start_epoch_time)/60
        duration = torch.Tensor([duration]).to(rank_local)
        dist.reduce(duration, dst=0, op=dist.ReduceOp.SUM)
        duration = duration.tolist()

        if rank_global == 0:
            print(f"Training Time for epoch {epoch+1} = {duration[0]/world_size} minutes")

        print(f"Running validation for epoch {epoch+1}...")
        # Do validation
        rec.eval()

        with torch.no_grad():
            batch_iter_val = dataloader.prepare_batches_prefetch(ratings_val, movies_dataset, rank_local, 5, device=rank_local, prefetch_factor=0)
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
                
                    batch_loss:torch.Tensor = criterion(output.cpu().contiguous(), labels.cpu().contiguous())

                    sum_loss += output.shape[0]*batch_loss.item()
                    sum_rows += output.shape[0]

                    if rank_global == 0:
                        print(f"Epoch: {epoch+1}, Batch: {i+1}, Loss (Rank 0): {sum_loss/sum_rows}")

                    # Clear cache after each 10 batches
                    if (i+1) % 10:
                        torch.cuda.empty_cache()
                        gc.collect()

                except StopIteration:
                    break

                i += 1

                if i >= max_num_batches:
                    break
            
            # Compute average validation loss after 1st epoch
            vloss = torch.Tensor([sum_loss, sum_rows]).to(rank_local)
            dist.reduce(vloss, dst=0, op=dist.ReduceOp.SUM)

            vloss = vloss.tolist()
            avg_vloss = vloss[0]/vloss[1]

            if rank_global == 0:
                print(f"Average Validation Loss: {avg_vloss}")
                print()
            
            # Checkpoint only through rank=0 worker because same weights across all workers after sync
            if rank_global == 0:
                print("Checkpointing...")
                
                if avg_vloss < best_vloss:
                    best_vloss = avg_vloss
                    checkpoint(rec.module, optimizer, os.path.join(model_out_dir, f"checkpoint-best-vloss.pth"))
    
    model:RecommenderSystem = rec.module

    # Save final model by rank=0 worker
    if rank_global == 0:
        print("Saving model...")
        checkpoint(model, optimizer, os.path.join(model_out_dir, f"final_model.pth"))

        print("Uploading model to GCS...")
        upload_directory_with_transfer_manager(
            gcs_bucket_name, 
            model_out_dir, 
            f"{gcs_prefix}/{model_out_dir}"
        )

    # Save movie embeddings and user embeddings
    save_embeddings_and_metadata(
        model,
        datasets_gcs_path,
        movies_dataset,
        gcs_bucket_name,
        gcs_prefix,
        world_size,
        rank_global,
        rank_local
    )

    # except Exception as e:
    #     print(e)
    # finally:
    #     # destroy ddp processes
    #     if dist.is_initialized():
    #         dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ray-based UEM Model Training')

    parser.add_argument('--gcs_bucket', type=str, required=True,
                        help='GCS Bucket')
    parser.add_argument('--gcs_prefix', type=str, required=True,
                        help='GCS Prefix')
    parser.add_argument('--gcs_data_dir', type=str, required=True,
                        help='Path to GCS data directory')
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
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers to prefetch batches')
    parser.add_argument('--model_out_dir', type=str, default=None,
                        help='Model output directory LOCAL')

    args = parser.parse_args()

    train_func(vars(args))

    """
    nohup torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=8 \
        trainer.py \
            --gcs_bucket "r6-ae-dev-adperf-adintelligence-data" \
            --gcs_prefix "amondal"  \
            --gcs_data_dir "parquet_dataset_ml_32m" \
            --batch_size 128 \
            --num_epochs 10 \
            --num_workers 4 \
            --accumulate_grad_batches 4 \
            --model_out_dir "/tmp/model_outputs" >output.log 2>&1 &



    torchrun \
        --nnodes=2 \
        --node_rank=0 \
        --master_addr=240.76.3.7 \
        --master_port=29500 \
        --nproc_per_node=8 \
        trainer.py \
            --gcs_bucket "r6-ae-dev-adperf-adintelligence-data" \
            --gcs_prefix "amondal"  \
            --gcs_data_dir "parquet_dataset_ml_32m" \
            --batch_size 128 \
            --max_num_batches 100 \
            --num_epochs 10 \
            --accumulate_grad_batches 4 \
            --model_out_dir "/tmp/model_outputs"

    torchrun \
        --nnodes=2 \
        --node_rank=1 \
        --master_addr=240.76.3.7 \
        --master_port=29500 \
        --nproc_per_node=8 \
        trainer.py \
            --gcs_bucket "r6-ae-dev-adperf-adintelligence-data" \
            --gcs_prefix "amondal"  \
            --gcs_data_dir "parquet_dataset_ml_32m" \
            --batch_size 128 \
            --num_epochs 10 \
            --accumulate_grad_batches 4 \
            --model_out_dir "/tmp/model_outputs"
    """