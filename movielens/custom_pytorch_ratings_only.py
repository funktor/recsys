import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import os
import sys
import joblib
from prepare_training_data import ratings_only_datasets
from custom_datasets import MovieLensRatingsOnlyDataset

base_dir = os.path.abspath(os.path.dirname(__file__))

def log_progress(epoch, n_epochs, step, avg_loss, data_size):
    sys.stderr.write(f"\r{epoch+1:02d}/{n_epochs:02d} | Step: {step}/{data_size} | Batch Loss: {avg_loss:<6.9f}")
    sys.stderr.flush()
    
class RecSysModel(nn.Module):
    def __init__(
        self,
        num_users,
        num_movies,
        embedding_size=256,
        hidden_dim=256,
        dropout_rate=0.2,
    ):
        super(RecSysModel, self).__init__()
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.hidden_dim = hidden_dim

        self.user_embedding = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_size)
        self.movie_embedding = nn.Embedding(num_embeddings=self.num_movies, embedding_dim=self.embedding_size)

        self.fc1 = nn.Linear(2 * self.embedding_size, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, 1)

        self.dropout = nn.Dropout(p=dropout_rate)

        self.relu = nn.ReLU()

    def forward(self, users, movies):
        user_embedded = self.user_embedding(users)
        movie_embedded = self.movie_embedding(movies)

        combined = torch.cat([user_embedded, movie_embedded], dim=1)

        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        output = self.fc2(x)

        return output
    
class RecSysTrainer:
    def __init__(self, num_users, num_movies, embedding_size=32, hidden_dim_size=32, dropout_rate=0.1, batch_size=256, n_workers=8, n_epochs=2, device='mps'):
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.hidden_dim_size = hidden_dim_size
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.n_epochs = n_epochs
        self.device = device
        self.model = None


    def train(self, train_dataset, validation_dataset):
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers, persistent_workers=True)
        validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size, num_workers=self.n_workers, persistent_workers=True)

        self.model = RecSysModel(
            num_users=self.num_users, 
            num_movies=self.num_movies,
            embedding_size=self.embedding_size,
            hidden_dim=self.hidden_dim_size,
            dropout_rate=self.dropout_rate,
        ).to(device=self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        log_progress_step = 128
        train_dataset_size = len(train_dataset)

        print(f"Training on {train_dataset_size} samples...")

        for e in range(self.n_epochs):
            train_loss = 0
            self.model.train(True)
            step_count = 0

            for i, train_data in enumerate(train_loader):
                output = self.model(train_data[0].to(device=self.device), train_data[1].to(device=self.device))
                output = output.squeeze()
                ratings = (train_data[2].to(torch.float32).to(device=self.device))

                loss = nn.functional.mse_loss(output, ratings, reduction='sum')
                train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                step_count += len(train_data[0])

                if (step_count % log_progress_step == 0 or i == len(train_loader) - 1):
                    log_progress(e, self.n_epochs, step_count, loss/len(train_data[0]), train_dataset_size)
            
            print()
            print(f"Training loss after epoch {e+1} = {train_loss/train_dataset_size}")

            
            validation_loss = 0
            self.model.train(False)
            step_count = 0
            for i, validation_data in enumerate(validation_loader):
                output = self.model(validation_data[0].to(device=self.device), validation_data[1].to(device=self.device))
                output = output.squeeze()
                ratings = (validation_data[2].to(torch.float32).to(device=self.device))

                loss = nn.functional.mse_loss(output, ratings, reduction='sum')
                validation_loss += loss.item()
                step_count += len(validation_data[0])
            
            print(f"Validation loss after epoch {e+1} = {validation_loss/len(validation_dataset)}")
            print()


    def evaluate(self, test_dataset):
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.n_workers, persistent_workers=True)

        if self.model is not None:
            test_loss = 0
            self.model.train(False)
            step_count = 0

            for i, test_data in enumerate(test_loader):
                output = self.model(test_data[0].to(device=self.device), test_data[1].to(device=self.device))
                output = output.squeeze()
                ratings = (test_data[2].to(torch.float32).to(device=self.device))

                loss = nn.functional.mse_loss(output, ratings, reduction='sum')
                test_loss += loss.item()
                step_count += len(test_data[0])
            
            print(f"Testing loss = {test_loss/len(test_dataset)}")
            print()


    def predict(self, predict_dataset):
        predict_loader = DataLoader(predict_dataset, batch_size=self.batch_size, num_workers=self.n_workers, persistent_workers=True)

        if self.model is not None:
            predict_loss = 0
            self.model.train(False)
            step_count = 0

            actuals, preds = [], []

            for i, predict_data in enumerate(predict_loader):
                output = self.model(predict_data[0].to(device=self.device), predict_data[1].to(device=self.device))
                output = output.squeeze()
                ratings = (predict_data[2].to(torch.float32).to(device=self.device))

                actuals += ratings.tolist()
                preds += output.tolist()

                loss = nn.functional.mse_loss(output, ratings, reduction='sum')
                predict_loss += loss.item()
                step_count += len(predict_data[0])
            
            print(f"Prediction loss = {predict_loss/len(predict_dataset)}")
            print()
            return actuals, preds
    
        return None
    
if __name__ == '__main__':
    out_dir = os.path.join(base_dir, 'training_data')
    ratings_only_datasets()
    n_users, n_movies = joblib.load(os.path.join(out_dir, 'num_users_movies.pkl'))

    train_dataset = MovieLensRatingsOnlyDataset(data_path=os.path.join(out_dir, 'train.csv'))
    test_dataset = MovieLensRatingsOnlyDataset(data_path=os.path.join(out_dir, 'test.csv'))
    validation_dataset = MovieLensRatingsOnlyDataset(data_path=os.path.join(out_dir, 'validation.csv'))

    trainer = RecSysTrainer(n_users, n_movies)
    trainer.train(train_dataset, validation_dataset)
    trainer.evaluate(test_dataset)
    actuals, preds = trainer.predict(test_dataset)

    print(list(zip(actuals, preds)))