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

class MatrixFactorization(L.LightningModule):
    def __init__(self, num_users, num_items, embedding_dim):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, user_id, item_id):
        user_vector = self.user_embedding(user_id)
        item_vector = self.item_embedding(item_id)
        return (user_vector * item_vector).sum(1)

    def training_step(self, batch, batch_idx):
        user_id, item_id, rating = batch
        prediction = self(user_id, item_id)
        loss = nn.functional.mse_loss(prediction, rating)
        self.log(name='train_loss', value=loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        user_id, item_id, rating = batch
        prediction = self(user_id, item_id)
        loss = nn.functional.mse_loss(prediction, rating)
        self.log(name='val_loss', value=loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        user_id, item_id, rating = batch
        prediction = self(user_id, item_id)
        loss = nn.functional.mse_loss(prediction, rating)
        self.log(name='test_loss', value=loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss
    
    def prediction_step(self, batch, batch_idx):
        user_id, item_id, rating = batch
        prediction = self(user_id, item_id)
        loss = nn.functional.mse_loss(prediction, rating)
        self.log(name='prediction_loss', value=loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)
    
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
        self.facto = None


    def train(self, train_dataset, validation_dataset):
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers, persistent_workers=True)
        validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size, num_workers=self.n_workers, persistent_workers=True)

        self.facto = MatrixFactorization(self.num_users, self.num_movies, embedding_dim=self.embedding_size)
        self.model = L.Trainer(max_epochs=self.n_epochs, log_every_n_steps=128, accelerator=self.device)
        self.model.fit(self.facto, train_dataloaders=train_loader, val_dataloaders=validation_loader)


    def evaluate(self, test_dataset):
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.n_workers, persistent_workers=True)

        if self.model is not None:
            res = self.model.test(self.facto, dataloaders=test_loader)
            print(res)


    def predict(self, predict_dataset):
        predict_loader = DataLoader(predict_dataset, batch_size=self.batch_size, num_workers=self.n_workers, persistent_workers=True)

        if self.model is not None:
            actuals = []
            for predict_data in predict_loader:
                ratings = (predict_data[2].to(torch.float32).to(device=self.device))
                actuals += ratings.tolist()

            preds = self.model.predict(self.facto, dataloaders=predict_loader)
            print(actuals)
            print(preds)
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