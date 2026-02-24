import argparse
import os
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.strategies import DDPStrategy
from mlflow.tracking import MlflowClient
from torch.utils.data import DataLoader, IterableDataset

import ray
from ray.train import ScalingConfig, RunConfig, Checkpoint
from ray.train.torch import TorchTrainer
from ray.train import get_dataset_shard
import ray.train as train
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
import pytorch_lightning as pl

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = None
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

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


def get_trainer_and_optimizer(vocabulary:dict, device):
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
        model.RecommenderSystem(
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
        ).to(device=device)
    
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
    
    
def train_func(config: dict):
    world_size = ray.train.get_context().get_world_size()
    rank = ray.train.get_context().get_local_rank()

    datasets_gcs_path = config["gcs_dir"]
    ratings_train_path = f"{datasets_gcs_path}/train/*.parquet"
    path_vocab = f"{datasets_gcs_path}/vocabulary.pkl"
    batch_size = config["batch_size"]
    max_num_epochs = config["num_epochs"]
    accumulate_grad_batches = config["accumulate_grad_batches"]

    ratings_train_mmap, ratings_val_mmap, movies_dataset, n_train, m_train, cols_train, n_val, m_val, cols_val = dataloader.get_datasets(datasets_gcs_path)

    num_train_data = count_rows_in_gcs_parquet(ratings_train_path)
    batches_per_epoch = num_train_data // (world_size*batch_size)

    vocabulary = dataloader.get_vocabulary(path_vocab, f"/tmp/{rank}/vocabulary.pkl")

    rec, optimizer = get_trainer_and_optimizer(vocabulary, device)
    optimizer.zero_grad()

    criterion = nn.MSELoss()
    scheduler = CosineWarmupScheduler(optimizer, warmup=50, max_iters=batches_per_epoch*max_num_epochs/accumulate_grad_batches)

    for epoch in range(max_num_epochs):
        rec.train()
        batch_iter = dataloader.prepare_batches(ratings_train_mmap, movies_dataset, n_train, m_train, cols_train, batch_size, device=device, batch_limit=batches_per_epoch)
        i = 0
        while True:
            batch = next(batch_iter)

            if batch:
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
                
                loss:torch.Tensor = criterion(output.contiguous(), labels.contiguous())
                loss /= accumulate_grad_batches
                loss.backward()

                if (i+1) % accumulate_grad_batches == 0:
                    nn.utils.clip_grad_norm_(rec.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    print(f"Epoch: {epoch+1}, Batch: {i+1}, Loss: {loss.item()}")
            else:
                break

            i += 1

        rec.eval()
        batch_iter_val = dataloader.prepare_batches(ratings_val_mmap, movies_dataset, n_val, m_val, cols_val, batch_size, device=device)
        sum_loss = 0.0

        i = 0
        while True:
            batch = next(batch_iter_val)

            if batch:
                data, labels = batch
                user_ids, user_prev_rated_movie_ids, user_prev_ratings, movie_ids, movie_descriptions, movie_genres, movie_years = data

                with torch.no_grad():
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
                
                    loss:torch.Tensor = criterion(output.contiguous(), labels.contiguous())
                    sum_loss += output.shape[0]*loss.item()
            else:
                break

            i += 1
        
        vloss = sum_loss/n_val
        print(f"Validation Loss: {vloss}")
        print()


def main(args: argparse.Namespace):
    args_dict = vars(args)

    if not ray.is_initialized():
        ray.init(address=os.environ.get('RAY_ADDRESS', 'auto'))

    # Load data using Ray Data if specified
    if args_dict.get('use_ray_data', False):
        print("Loading training data with Ray Data...")
        train_dataset = ray.data.read_parquet(
            args_dict['train_dir'],
            override_num_blocks=args_dict.get('num_workers', 4) * 4,
        )
        val_dataset = ray.data.read_parquet(
            args_dict['val_dir'],
            override_num_blocks=args_dict.get('num_workers', 4) * 2,
        )

        print(f"Train dataset: {train_dataset.count()} rows")
        print(f"Val dataset: {val_dataset.count()} rows")

        datasets_dict = {"train": train_dataset, "val": val_dataset}
    else:
        # Lightning DataModule will handle data loading
        datasets_dict = None

    # Configure Ray TorchTrainer
    num_workers = args_dict.get('num_workers', 8)

    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=True,
        resources_per_worker={
            "CPU": 8,
            "GPU": 1,
        },
    )

    run_config = RunConfig(
        name=f"uem_training_{args_dict['run_name']}",
        storage_path=f"{args_dict['checkpoint_dir']}/ray_results/",
        checkpoint_config=train.CheckpointConfig(
            num_to_keep=3,
            checkpoint_score_attribute="final_val_auc",
            checkpoint_score_order="max",
        ),
        failure_config=train.FailureConfig(
            max_failures=3,  # Allow up to 3 worker failures before giving up
            fail_fast=False,  # Continue training even if a worker fails
        ),
    )

    # Create TorchTrainer - NO @ray.remote decorator needed!
    print("Creating TorchTrainer...")
    ray_trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=args_dict,
        scaling_config=scaling_config,
        datasets=datasets_dict,
        run_config=run_config,
    )

    print("Starting Ray TorchTrainer...")
    ray_trainer.fit()
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ray-based UEM Model Training')

    # Data paths
    parser.add_argument('--vocabulary_path', type=str, required=True,
                        help='Path to vocabulary file (local or GCS)')
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Path to training parquet files')
    parser.add_argument('--val_dir', type=str, required=True,
                        help='Path to validation parquet files')
    parser.add_argument('--test_dir', type=str, default=None,
                        help='Path to test parquet files')
    parser.add_argument('--auc_data_dir', type=str, default=None,
                        help='Path to AUC evaluation data')
    parser.add_argument('--score_dir', type=str, default=None,
                        help='Path to writing test scores')

    # MLflow configuration
    parser.add_argument('--mlflow_uri', type=str, required=True,
                        help='MLflow tracking URI')
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='MLflow experiment name')
    parser.add_argument('--run_name', type=str, required=True,
                        help='MLflow run name')
    parser.add_argument('--model_alias', type=str, default='champion_candidate',
                        help='Model alias for MLflow registry')

    # Training configuration
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size per worker')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of Ray workers for distributed training')
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='Use GPU for training (default: True)')
    parser.add_argument('--max_epochs', type=int, default=40,
                        help='Maximum number of training epochs')
    parser.add_argument('--accumulate_grad_batches', type=int, default=4,
                        help='Accumulate gradients over N batches')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')

    # Data loading strategy
    parser.add_argument('--use_ray_data', action='store_true', default=False,
                        help='Use Ray Data for distributed data loading (default: Lightning DataModule)')

    # Checkpoint configuration
    parser.add_argument('--checkpoint_dir', type=str, default="/home/jarvis/checkpoints",
                        help='Directory to save checkpoints')
    parser.add_argument('--prev_model_dir', type=str, default=None,
                        help='Path to previous model checkpoint to resume training')

    # Other
    parser.add_argument('--gendata_run_id', type=str, required=True,
                        help='Run ID for cloud path storing run-related data')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Enable debug mode with verbose logging')

    args = parser.parse_args()

    # Run training
    main(args)
