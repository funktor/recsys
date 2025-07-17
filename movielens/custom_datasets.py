import torch
import pandas as pd
from torch.utils.data import Dataset

class MovieLensRatingsOnlyDataset(Dataset):
    def __init__(self, data_path=None, dataframe=None):
        self.dataframe = dataframe or self.load_movielens_data(data_path)
    
    @staticmethod
    def load_movielens_data(data_path):
        column_names = ['userId', 'movieId', 'rating', 'timestamp']
        df = pd.read_csv(data_path, sep=',', names=column_names, dtype={'userId':'int32', 'movieId':'int32', 'rating':float, 'timestamp':'int64'}, header=0)
        df.dropna(inplace=True)

        df['userId'] -= df['userId'].min()
        df['movieId'] -= df['movieId'].min()
        return df

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        user_id = self.dataframe.iloc[idx, 0]
        item_id = self.dataframe.iloc[idx, 1]
        rating = self.dataframe.iloc[idx, 2]
        return (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(item_id, dtype=torch.long),
            torch.tensor(rating, dtype=torch.float)
        )