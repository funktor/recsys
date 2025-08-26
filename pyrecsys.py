import math
import os
import numpy as np
import pandas as pd
import random
import uuid
import joblib
import pytorch_lightning as pl

import torch
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
    
print("Device:", device)

def checkpoint(model:nn.Module, optimizer:torch.optim.Optimizer, filename):
    torch.save({'optimizer':optimizer.state_dict(), 'model':model.state_dict()}, filename)

    
def load_model(filename):
    chkpt = torch.load(filename, weights_only=False)
    return chkpt['model'], chkpt['optimizer']


def attention(q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, mask=None):
    d_k = q.size()[-1] # q,k,v : (batch, head, seq_len, embed_size_per_head)
    attn_logits = torch.matmul(q, k.transpose(-2, -1)) # (batch, head, seq_len, seq_len)
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v) # (batch, head, seq_len, embed_size_per_head)
    return values, attention


def init_weights(x:nn.Linear):
    with torch.no_grad():
        nn.init.xavier_uniform_(x.weight)
        x.bias.data.fill_(0)
		
        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model) # (seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model)) # (seq_len, d_model)
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model)) # (seq_len, d_model)
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x:torch.Tensor):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)   
        return self.dropout(x)
	
    
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, input_dim:int, d_model: int, h: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h

        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h

        self.w_q = nn.Linear(input_dim, d_model) # Wq
        self.w_k = nn.Linear(input_dim, d_model) # Wk
        self.w_v = nn.Linear(input_dim, d_model) # Wv
        self.w_o = nn.Linear(d_model, d_model) # Wo

        init_weights(self.w_q)
        init_weights(self.w_k)
        init_weights(self.w_v)
        init_weights(self.w_o)

    def forward(self, q_x:torch.Tensor, k_x:torch.Tensor, v_x:torch.Tensor, mask=None):
        q:torch.Tensor = self.w_q(q_x) # (batch, seq_len, d_model)
        k:torch.Tensor = self.w_k(k_x) # (batch, seq_len, d_model)
        v:torch.Tensor = self.w_v(v_x) # (batch, seq_len, d_model)

        q_h = q.reshape(q.shape[0], q.shape[1], self.h, self.d_k).transpose(1, 2) # (batch, head, seq_len, d_k)
        k_h = k.reshape(k.shape[0], k.shape[1], self.h, self.d_k).transpose(1, 2) # (batch, head, seq_len, d_k)
        v_h = v.reshape(v.shape[0], v.shape[1], self.h, self.d_k).transpose(1, 2) # (batch, head, seq_len, d_k)

        attn_out, _ = attention(q_h, k_h, v_h, mask) # (batch, head, seq_len, embed_size_per_head)
        attn_out = attn_out.transpose(1, 2) # (batch, seq_len, head, embed_size_per_head)
        attn_out = attn_out.reshape(attn_out.shape[0], attn_out.shape[1], attn_out.shape[2]*attn_out.shape[3]) # (batch, seq_len, d_model)

        return self.w_o(attn_out) # (batch, seq_len, d_model)
    
    
class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        super().__init__()

        self.self_attn = MultiHeadAttentionBlock(input_dim, input_dim, num_heads)

        self.ffn_1 = nn.Linear(input_dim, dim_feedforward)
        self.ffn_2 = nn.Linear(dim_feedforward, input_dim)

        init_weights(self.ffn_1)
        init_weights(self.ffn_2)

        self.ffn = nn.Sequential(
            self.ffn_1,
            nn.Dropout(dropout),
            nn.GELU(),
            self.ffn_2,
        )

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, x, x, mask=mask) # (batch, seq_len, input_dim)
        x = x + self.dropout(attn_out) # (batch, seq_len, input_dim)
        x = self.norm1(x) # (batch, seq_len, input_dim)

        ffn_out = self.ffn(x) # (batch, seq_len, input_dim)
        x = x + self.dropout(ffn_out) # (batch, seq_len, input_dim)
        x = self.norm2(x) # (batch, seq_len, input_dim)

        return x
	
    
class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dim_feedforward, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(d_model, num_heads, dim_feedforward, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x
	
    
class CrossFeatureLayer(nn.Module):
    def __init__(self, input_dim, num_layers,dropout=0.0) -> None:
        super(CrossFeatureLayer, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.cross_layer_params = []
        self.cross_layer_norms = []
        
        for i in range(num_layers):
            h = nn.Linear(input_dim, input_dim)
            init_weights(h)
            self.cross_layer_params += [h]

            g = nn.Sequential(
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(input_dim)
            )

            self.cross_layer_norms += [g]

        self.cross_layer_params = nn.ModuleList(self.cross_layer_params)
        self.cross_layer_norms = nn.ModuleList(self.cross_layer_norms)

    def forward(self, x):
        x_initial = torch.Tensor(x) # (batch, ..., input_dim)

        for i in range(self.num_layers):
            x = x_initial*self.cross_layer_params[i](x) + x # (batch, ..., input_dim)
            x = self.cross_layer_norms[i](x) # (batch, ..., input_dim)
        
        return x
    
    
class MovieEncoder(nn.Module):
    def __init__(
            self, 
            movie_vocab_size, 
            genres_vocab_size, 
            years_vocab_size, 
            embedding_size, 
            dropout=0.0
        ) -> None:
        
        super(MovieEncoder, self).__init__()
        
        self.movie_embedding_layer = nn.Embedding(movie_vocab_size, embedding_size)
        self.years_embedding_layer = nn.Embedding(years_vocab_size, 32)

        self.genres_encoder_layer = nn.Linear(genres_vocab_size, 4)
        init_weights(self.genres_encoder_layer)

        self.fc_concat = nn.Linear(embedding_size + 36, embedding_size)
        init_weights(self.fc_concat)

        self.fc = nn.Sequential(
            self.fc_concat,
            nn.GELU()
        )

        self.cross_layer = nn.Linear(embedding_size, embedding_size)
        init_weights(self.cross_layer)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embedding_size)

        self.cross_features = CrossFeatureLayer(embedding_size + 36, 3, dropout)

    def forward(self, movies, genres, years):
        movie_embedding = self.movie_embedding_layer(movies) # (batch, ..., embedding_size)
        genres_embedding = self.genres_encoder_layer(genres) # (batch, ..., 4)
        years_embedding = self.years_embedding_layer(years) # (batch, ..., 32)

        movie_embedding = torch.concat([movie_embedding, genres_embedding, years_embedding], dim=-1) # (batch, ..., embedding_size + 36)
        movie_embedding = self.cross_features(movie_embedding) + movie_embedding
        movie_embedding = self.fc(movie_embedding) # (batch, ..., embedding_size)
        movie_embedding = self.dropout(movie_embedding) # (batch, ..., embedding_size)
        movie_embedding = self.norm(movie_embedding) # (batch, ..., embedding_size)

        return movie_embedding
	
    
class UserEncoder(nn.Module):
    def __init__(
            self, 
            user_vocab_size, 
            movie_vocab_size, 
            genres_vocab_size, 
            years_vocab_size, 
            embedding_size, 
            movie_seq_len, 
            num_encoder_layers, 
            num_heads, 
            dim_ff,
            dropout=0.0
        ) -> None:

        super(UserEncoder, self).__init__()
        
        self.embedding_size = embedding_size
        self.movie_seq_len = movie_seq_len
        self.num_encoder_layers = num_encoder_layers
        self.num_heads = num_heads
        self.dim_ff = dim_ff

        self.user_embedding_layer = nn.Embedding(user_vocab_size, embedding_size)
        self.movie_encoder = MovieEncoder(movie_vocab_size, genres_vocab_size, years_vocab_size, embedding_size, dropout)

        self.positional_encoding = PositionalEncoding(embedding_size, movie_seq_len, dropout)
        self.encoder_block = Encoder(num_encoder_layers, embedding_size, num_heads, dim_ff, dropout)

        self.fc_concat = nn.Linear(2*embedding_size, embedding_size)
        init_weights(self.fc_concat)

        self.fc = nn.Sequential(
            self.fc_concat,
            nn.GELU()
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embedding_size)


    def forward(
            self, 
            user_ids, 
            rated_movie_ids, 
            rated_movie_genres, 
            rated_movie_years, 
            rated_movie_ratings):
        
        user_embedding = self.user_embedding_layer(user_ids) # (batch, 1, embedding_size)
        
        movie_embeddings = self.movie_encoder(rated_movie_ids, rated_movie_genres, rated_movie_years) # (batch, movie_seq_len, embedding_size)
        movie_embeddings = self.positional_encoding(movie_embeddings) # (batch, movie_seq_len, embedding_size)
        movie_embeddings = self.encoder_block(movie_embeddings, None) # (batch, movie_seq_len, embedding_size)

        rated_movie_ratings = F.softmax(rated_movie_ratings, dim=-1) # (batch, movie_seq_len)
        rated_movie_ratings = rated_movie_ratings.unsqueeze(1) # (batch, 1, movie_seq_len)
        movie_embeddings_weighted = torch.matmul(rated_movie_ratings, movie_embeddings) # (batch, 1, embedding_size)

        user_embedding = torch.concat([user_embedding, movie_embeddings_weighted], dim=-1) # (batch, 1, 2*embedding_size)
        user_embedding = self.fc(user_embedding) # (batch, 1, embedding_size)
        user_embedding = self.dropout(user_embedding) # (batch, 1, embedding_size)
        user_embedding = self.norm(user_embedding) # (batch, 1, embedding_size)

        return user_embedding, movie_embeddings
	
    
class RecommenderSystem(nn.Module):
    def __init__(
            self, 
            user_vocab_size, 
            movie_vocab_size, 
            genres_vocab_size, 
            years_vocab_size, 
            embedding_size, 
            movie_seq_len, 
            num_encoder_layers, 
            num_heads, 
            dim_ff,
            dropout=0.0
        ) -> None:

        super(RecommenderSystem, self).__init__()

        self.embedding_size = embedding_size
        self.movie_seq_len = movie_seq_len
        self.num_encoder_layers = num_encoder_layers
        self.num_heads = num_heads
        self.dim_ff = dim_ff

        self.movie_encoder = \
            MovieEncoder\
            (
                movie_vocab_size, 
                genres_vocab_size, 
                years_vocab_size, 
                embedding_size, 
                dropout
            )
        
        self.user_encoder = \
            UserEncoder\
            (
                user_vocab_size, 
                movie_vocab_size, 
                genres_vocab_size, 
                years_vocab_size, 
                embedding_size, 
                movie_seq_len, 
                num_encoder_layers, 
                num_heads, 
                dim_ff,
                dropout
            )
        
        self.cross_attn = MultiHeadAttentionBlock(embedding_size, embedding_size, num_heads)

        self.fc_concat1 = nn.Linear(2*embedding_size, embedding_size)
        init_weights(self.fc_concat1)

        self.fc1 = nn.Sequential(
            self.fc_concat1,
            nn.GELU()
        )

        self.fc_concat2 = nn.Linear(2*embedding_size, embedding_size)
        init_weights(self.fc_concat2)

        self.fc2 = nn.Sequential(
            self.fc_concat2,
            nn.GELU()
        )

        self.fc_ratings_linear = nn.Linear(embedding_size, 1)
        init_weights(self.fc_ratings_linear)

        self.fc_ratings = nn.Sequential(
            self.fc_ratings_linear,
            nn.ReLU(inplace=True)
        )

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.norm3 = nn.LayerNorm(embedding_size)


    def forward(
            self, 
            user_ids, 
            movie_ids, 
            rated_movie_ids, 
            rated_movie_genres, 
            rated_movie_years, 
            rated_movie_ratings, 
            movie_genres, 
            movie_years):
        
        movie_embeddings = \
            self.movie_encoder(movie_ids, movie_genres, movie_years) # (batch, 1, embedding_size)
        
        user_embeddings, rated_movie_embeddings = \
            self.user_encoder\
                (
                    user_ids, 
                    rated_movie_ids, 
                    rated_movie_genres, 
                    rated_movie_years, 
                    rated_movie_ratings
                )                     # (batch, 1, embedding_size), (batch, movie_seq_len, embedding_size)
        
        crss_attn_out = self.cross_attn(movie_embeddings, rated_movie_embeddings, rated_movie_embeddings, mask=None) # (batch, 1, embedding_size)
        rated_movie_embeddings = movie_embeddings + self.dropout(crss_attn_out) # (batch, 1, embedding_size)
        rated_movie_embeddings = self.norm1(rated_movie_embeddings) # (batch, 1, embedding_size)

        movie_embeddings = torch.concat([movie_embeddings, rated_movie_embeddings], dim=-1) # (batch, 1, 2*embedding_size)
        movie_embeddings = self.fc1(movie_embeddings) # (batch, 1, embedding_size)
        movie_embeddings = self.dropout(movie_embeddings) # (batch, 1, embedding_size)
        movie_embeddings = self.norm2(movie_embeddings) # (batch, 1, embedding_size)
        
        encoded = torch.concat([user_embeddings, movie_embeddings], dim=-1) # (batch, 1, 2*embedding_size)
        encoded = self.fc2(encoded) # (batch, 1, embedding_size)
        encoded = self.dropout(encoded) # (batch, 1, embedding_size)
        encoded = self.norm3(encoded) # (batch, 1, embedding_size)

        output = self.fc_ratings(encoded) # (batch, 1, 1)
        output = torch.clamp(output, min=0.0, max=5.0).squeeze(1) # (batch, 1)

        return output
	
    
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

    
def read_ml_32m_data(folder):
    print("Reading movies and ratings data...")
    # Read ratings and movies data from folder
    ratings_path = os.path.join(folder, 'ratings.csv')
    genres_path = os.path.join(folder, 'movies.csv')

    rating_column_names = ['userId', 'movieId', 'rating', 'timestamp']
    genres_column_names = ['movieId', 'title', 'genres']

    df_rating = pd.read_csv(ratings_path, sep=',', names=rating_column_names, dtype={'userId':'int32', 'movieId':'int32', 'rating':float, 'timestamp':'int64'}, header=0)
    df_genres = pd.read_csv(genres_path, sep=',', names=genres_column_names, dtype={'movieId':'int32', 'title':'object', 'genres':'object'}, header=0)

    df_rating.dropna(inplace=True, subset=['userId', 'movieId', 'rating'])
    df_genres.dropna(inplace=True, subset=['movieId', 'title', 'genres'])

    # Extract movie genres
    df_genres['genres'] = df_genres['genres'].apply(lambda x: x.split('|'))

    # Extract movie year from title
    df_genres['movie_year'] = df_genres['title'].str.extract(r'\((\d{4})\)').fillna("2025").astype('int')
    df_genres.drop(columns=['title'], inplace=True)

    df = df_rating.merge(df_genres, on=['movieId'], how='left')

    return df


def ohe_genres(df:pd.DataFrame):
    print("Doing one hot encoding of genres...")
    # One hot encoding of genres
    all_genres = df['genres'].tolist()
    genres_set = set()

    for x in all_genres:
        genres_set.update(set(x))

    genres_set = list(genres_set)
    inv_idx = {genres_set[i]:i for i in range(len(genres_set))}

    genres_mh = []
    for x in all_genres:
        h = [0]*len(genres_set)
        for y in x:
            h[inv_idx[y]] = 1
        genres_mh += [h]

    df['genres_mh'] = genres_mh
    df.drop(columns=['genres'], inplace=True)

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


def get_recsys_train_data(df:pd.DataFrame, prev_seq_len = 3, sample_movie_k=20):
	df2 = df.groupby(by=["userId"]).agg(list).reset_index()

	user_ids, movie_ids, genres, years, ratings = [], [], [], [], []
	
	prev_movie_ids = []
	prev_movie_genres = []
	prev_movie_years = []
	prev_movie_ratings = []

	for i in range(df2.shape[0]):
		movie_ids_seq = df2.loc[i, 'movieId']
		user_id = df2.loc[i, 'userId']
		genres_seq = df2.loc[i, 'genres_mh']
		ratings_seq = df2.loc[i, 'rating']
		years_seq = df2.loc[i, 'movie_year']

		m = len(movie_ids_seq)-prev_seq_len
		if m > 0:
			# Sample movie sequences for each user id
			indices = random.sample(range(prev_seq_len, len(movie_ids_seq)), k=min(m, sample_movie_k))

			for j in indices:
				rated_movie_ids = movie_ids_seq[max(0, j-prev_seq_len):j]
				rated_movie_genres = genres_seq[max(0, j-prev_seq_len):j]
				rated_movie_years = years_seq[max(0, j-prev_seq_len):j]
				rated_movie_ratings = ratings_seq[max(0, j-prev_seq_len):j]

				user_ids += [user_id]
				movie_ids += [movie_ids_seq[j]]
				genres += [genres_seq[j]]
				years += [years_seq[j]]
				ratings += [ratings_seq[j]]

				prev_movie_ids += [rated_movie_ids]
				prev_movie_genres += [rated_movie_genres]
				prev_movie_years += [rated_movie_years]
				prev_movie_ratings += [rated_movie_ratings]

	# Convert to Tensors
	user_ids = torch.tensor(user_ids, dtype=torch.int32)
	movie_ids = torch.tensor(movie_ids, dtype=torch.int32)
	genres = torch.tensor(genres, dtype=torch.int8)
	years = torch.tensor(years, dtype=torch.int32)
	ratings = torch.tensor(ratings, dtype=torch.float32)

	prev_movie_ids = torch.tensor(prev_movie_ids, dtype=torch.int32)
	prev_movie_genres = torch.tensor(prev_movie_genres, dtype=torch.int8)
	prev_movie_years = torch.tensor(prev_movie_years, dtype=torch.int32)
	prev_movie_ratings = torch.tensor(prev_movie_ratings, dtype=torch.float32)

	return \
		user_ids, \
		movie_ids, \
		genres, \
		years, \
		ratings, \
		prev_movie_ids, \
		prev_movie_genres, \
		prev_movie_years, \
		prev_movie_ratings


def get_model_and_optimizer(user_id_vocab_size, movie_id_vocab_size, genres_vocab_size, years_vocab_size):
    # Get model class and optimizer
    embedding_size = 128
    movie_seq_len = 10
    num_encoder_layers = 4
    num_heads = 4
    dropout = 0.1
    dff = 32

    rec = \
        RecommenderSystem\
        (
            user_id_vocab_size, 
            movie_id_vocab_size, 
            genres_vocab_size, 
            years_vocab_size, 
            embedding_size, 
            movie_seq_len, 
            num_encoder_layers, 
            num_heads, 
            dff, 
            dropout
        ).to(device=device)

    optimizer = optim.Adam(rec.parameters(), lr=0.001)
    return rec, optimizer


def train_and_validate_model(df_train:pd.DataFrame, df_val:pd.DataFrame, model_dir:str='models', run_id=1):
    print("Training model...")
    # Get vocabulary sizes
    user_id_vocab_size = int(df_train["userId"].max()+1)
    movie_id_vocab_size = int(df_train["movieId"].max()+1)
    genres_vocab_size = int(len(df_train["genres_mh"][0]))
    years_vocab_size = int(df_train["movie_year"].max()+1)

    (
        user_ids_train, 
        movie_ids_train, 
        genres_train, 
        years_train, 
        ratings_train, 
        prev_movie_ids_train, 
        prev_movie_genres_train, 
        prev_movie_years_train, 
        prev_movie_ratings_train

    ) = get_recsys_train_data(df_train)

    (
        user_ids_val, 
        movie_ids_val, 
        genres_val, 
        years_val, 
        ratings_val, 
        prev_movie_ids_val, 
        prev_movie_genres_val, 
        prev_movie_years_val, 
        prev_movie_ratings_val

    ) = get_recsys_train_data(df_val)

    mode_dir_path = os.path.join(model_dir, run_id)
    os.makedirs(mode_dir_path, exist_ok=True)

    # Save vocabulary for loading model
    joblib.dump((
        user_id_vocab_size, 
        movie_id_vocab_size, 
        genres_vocab_size, 
        years_vocab_size), 
    os.path.join(mode_dir_path, 'vocabulary.pkl'))

    rec, optimizer = \
        get_model_and_optimizer(user_id_vocab_size, movie_id_vocab_size, genres_vocab_size, years_vocab_size)

    n_epochs = 10    # number of epochs to run
    batch_size = 512  # size of each batch
    batches_per_epoch = user_ids_train.shape[0] // batch_size

    criterion = nn.MSELoss()
    lr_scheduler = CosineWarmupScheduler(optimizer, warmup=50, max_iters=batches_per_epoch*n_epochs)

    best_vloss = float("Inf")

    for epoch in range(n_epochs):
        indices = torch.randperm(user_ids_train.shape[0])

        rec.train()
        for i in range(batches_per_epoch):
            optimizer.zero_grad()
            start = i * batch_size
            batch_indices = indices[start:start+batch_size]

            user_ids_batch = user_ids_train[batch_indices].unsqueeze(1).to(device=device)
            movie_ids_batch = movie_ids_train[batch_indices].unsqueeze(1).to(device=device)
            genres_batch = genres_train[batch_indices].unsqueeze(1).to(dtype=torch.float32).to(device=device)
            years_batch = years_train[batch_indices].unsqueeze(1).to(device=device)
            ratings_batch = ratings_train[batch_indices].unsqueeze(1).to(device=device)

            prev_movie_ids_batch = prev_movie_ids_train[batch_indices].to(device=device)
            prev_movie_genres_batch = prev_movie_genres_train[batch_indices].to(dtype=torch.float32).to(device=device)
            prev_movie_ids_years = prev_movie_years_train[batch_indices].to(device=device)
            prev_movie_ids_ratings = prev_movie_ratings_train[batch_indices].to(device=device)

            output:torch.Tensor = \
                rec(
                    user_ids_batch, 
                    movie_ids_batch, 
                    prev_movie_ids_batch, 
                    prev_movie_genres_batch, 
                    prev_movie_ids_years, 
                    prev_movie_ids_ratings, 
                    genres_batch, 
                    years_batch
                )

            loss:torch.Tensor = criterion(output.contiguous(), ratings_batch.contiguous())

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            print(f"Epoch: {epoch+1}, Batch: {i+1}, Loss: {loss.item()}")

        rec.eval()
        s = 0.0
        for i in range(0, user_ids_val.shape[0], batch_size):
            batch_indices = list(range(i, min(i+batch_size, user_ids_val.shape[0])))

            user_ids_batch = user_ids_val[batch_indices].unsqueeze(1).to(device=device)
            movie_ids_batch = movie_ids_val[batch_indices].unsqueeze(1).to(device=device)
            genres_batch = genres_val[batch_indices].unsqueeze(1).to(dtype=torch.float32).to(device=device)
            years_batch = years_val[batch_indices].unsqueeze(1).to(device=device)
            ratings_batch = ratings_val[batch_indices].unsqueeze(1).to(device=device)

            prev_movie_ids_batch = prev_movie_ids_val[batch_indices].to(device=device)
            prev_movie_genres_batch = prev_movie_genres_val[batch_indices].to(dtype=torch.float32).to(device=device)
            prev_movie_ids_years = prev_movie_years_val[batch_indices].to(device=device)
            prev_movie_ids_ratings = prev_movie_ratings_val[batch_indices].to(device=device)

            with torch.no_grad():
                output:torch.Tensor = \
                    rec(
                        user_ids_batch, 
                        movie_ids_batch, 
                        prev_movie_ids_batch, 
                        prev_movie_genres_batch, 
                        prev_movie_ids_years, 
                        prev_movie_ids_ratings, 
                        genres_batch, 
                        years_batch
                    )

                loss:torch.Tensor = criterion(output.contiguous(), ratings_batch.contiguous())
                s += output.shape[0]*loss.item()

        vloss = s/user_ids_val.shape[0]
        print(f"Validation Loss: {vloss}")
        print()

        print("Checkpointing...")
        if vloss < best_vloss:
            best_vloss = vloss
            checkpoint(rec, optimizer, os.path.join(mode_dir_path, f"checkpoint-best-vloss.pth"))
    
    print("Saving model...")
    checkpoint(rec, optimizer, os.path.join(mode_dir_path, f"final_model.pth"))

    return rec
		
		
def test_model(df_test:pd.DataFrame, rec:nn.Module, batch_size = 1024, model_dir:str='models', run_id=1):
    print("Testing model...")
    # Check model performance on test data
    (
        user_ids_test, 
        movie_ids_test, 
        genres_test, 
        years_test, 
        ratings_test, 
        prev_movie_ids_test, 
        prev_movie_genres_test, 
        prev_movie_years_test, 
        prev_movie_ratings_test

    ) = get_recsys_train_data(df_test)

    criterion = nn.MSELoss()
    if rec is None:
        path = os.path.join(model_dir, run_id)
        if os.path.exists(path):
            print("Loading vocabulary...")
            (
                user_id_vocab_size, 
                movie_id_vocab_size, 
                genres_vocab_size, 
                years_vocab_size
            ) = joblib.load(os.path.join(path, 'vocabulary.pkl'))

            rec, _ = \
                get_model_and_optimizer(user_id_vocab_size, movie_id_vocab_size, genres_vocab_size, years_vocab_size)

            print("Loading model...")
            model_dict, _ = load_model(os.path.join(path, f"checkpoint-best-vloss.pth"))
            rec.load_state_dict(model_dict)

    rec.eval()
    s = 0.0

    # Using batches so as not to cause out of memory issues on device
    for i in range(0, user_ids_test.shape[0], batch_size):
        batch_indices = list(range(i, min(i+batch_size, user_ids_test.shape[0])))

        user_ids_batch = user_ids_test[batch_indices].unsqueeze(1).to(device=device)
        movie_ids_batch = movie_ids_test[batch_indices].unsqueeze(1).to(device=device)
        genres_batch = genres_test[batch_indices].unsqueeze(1).to(dtype=torch.float32).to(device=device)
        years_batch = years_test[batch_indices].unsqueeze(1).to(device=device)
        ratings_batch = ratings_test[batch_indices].unsqueeze(1).to(device=device)

        prev_movie_ids_batch = prev_movie_ids_test[batch_indices].to(device=device)
        prev_movie_genres_batch = prev_movie_genres_test[batch_indices].to(dtype=torch.float32).to(device=device)
        prev_movie_ids_years = prev_movie_years_test[batch_indices].to(device=device)
        prev_movie_ids_ratings = prev_movie_ratings_test[batch_indices].to(device=device)

        with torch.no_grad():
            output:torch.Tensor = \
                rec(
                    user_ids_batch, 
                    movie_ids_batch, 
                    prev_movie_ids_batch, 
                    prev_movie_genres_batch, 
                    prev_movie_ids_years, 
                    prev_movie_ids_ratings, 
                    genres_batch, 
                    years_batch
                )

            loss:torch.Tensor = criterion(output.contiguous(), ratings_batch.contiguous())
            s += output.shape[0]*loss.item()

    return s/user_ids_test.shape[0]
		
    
if __name__ == '__main__':
    run_id = str(uuid.uuid4())
    df = read_ml_32m_data('/home/abhijit/recsys/datasets/ml-32m')
    df = ohe_genres(df)
    df_train, df_val, df_test = split_train_test(df)
    rec = train_and_validate_model(df_train, df_val, run_id=run_id)
    print("Loss on Test data = ",test_model(df_test, rec, run_id=run_id))