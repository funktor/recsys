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
        
        for _ in range(num_layers):
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
    
class MovieId:
    movie_id_emb = None

    def __new__(cls, movie_id_size, emb_size=128):
        if cls.movie_id_emb is None:
            cls.movie_id_emb = nn.Embedding(movie_id_size, emb_size, padding_idx=0)
        return cls.movie_id_emb
    
def emb_averaging(inp:torch.Tensor, emb_layer:nn.Module, padding_idx:int=0):
    # inp : (batch, num_tokens)
    embeddings = emb_layer(inp) # (batch, num_tokens, emb_size)
    mask = (inp != padding_idx).float().unsqueeze(-1) # (batch, num_tokens, 1)
    masked_embeddings = embeddings * mask # (batch, num_tokens, emb_size)
    sum_embeddings = torch.sum(masked_embeddings, dim=1) # (batch, emb_size)
    sequence_lengths = torch.sum(mask, dim=1) # (batch, 1) # prevents division by zero by + 1
    sequence_lengths = torch.clamp(sequence_lengths, min=1.0)
    averaged_embeddings = sum_embeddings / sequence_lengths # (batch, emb_size)
    return averaged_embeddings

class MovieEncoder(nn.Module):
    def __init__(
            self, 
            movie_id_size, 
            movie_desc_size,
            movie_genres_size,
            movie_year_size, 
            embedding_size, 
            dropout=0.0
        ) -> None:
        
        super(MovieEncoder, self).__init__()
        
        self.movie_id_emb = MovieId(movie_id_size, 128)
        self.movie_desc_emb = nn.Embedding(movie_desc_size, 128, padding_idx=0)
        self.movie_genres_emb = nn.Embedding(movie_genres_size, 32, padding_idx=0)
        self.movie_year_emb = nn.Embedding(movie_year_size, 32, padding_idx=0)

        self.fc_concat = nn.Linear(320, embedding_size)
        init_weights(self.fc_concat)

        self.fc = nn.Sequential(
            self.fc_concat,
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embedding_size)
        )

        self.cross_features = CrossFeatureLayer(320, 3, 0.0)

    def forward(
            self, 
            ids:torch.Tensor, 
            descriptions:torch.Tensor, 
            genres:torch.Tensor, 
            years:torch.Tensor
        ):

        id_emb = self.movie_id_emb(ids) # (batch, 128)
        desc_emb = emb_averaging(descriptions, self.movie_desc_emb) # (batch, 128)
        genres_emb = emb_averaging(genres, self.movie_genres_emb) # (batch, 32)
        years_emb = self.movie_year_emb(years) # (batch, 32)

        movie_embedding = torch.concat([id_emb, desc_emb, genres_emb, years_emb], dim=-1) # (batch, 320)
        movie_embedding = self.cross_features(movie_embedding) + movie_embedding # (batch, 320)
        movie_embedding = self.fc(movie_embedding) # (batch, embedding_size)

        return movie_embedding
    
class UserEncoder(nn.Module):
    def __init__(
            self, 
            user_id_size, 
            movie_id_size,
            embedding_size, 
            prev_rated_seq_len, 
            num_encoder_layers, 
            num_heads=2, 
            dim_ff=128,
            dropout=0.0
        ) -> None:

        super(UserEncoder, self).__init__()

        self.user_id_emb = nn.Embedding(user_id_size, 128, padding_idx=0)
        self.movie_id_emb = MovieId(movie_id_size, 128)

        self.positional_encoding = PositionalEncoding(128, prev_rated_seq_len, 0.0)
        self.encoder_block = Encoder(num_encoder_layers, 128, num_heads, dim_ff, 0.0)

        self.fc_concat = nn.Linear(256, embedding_size)
        init_weights(self.fc_concat)

        self.fc = nn.Sequential(
            self.fc_concat,
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embedding_size)
        )

        self.num_heads = num_heads


    def forward(
            self, 
            user_ids:torch.Tensor, # (batch,)
            prev_rated_movie_ids:torch.Tensor, # (batch, seq_len)
            prev_ratings:torch.Tensor # (batch, seq_len)
        ):
        user_id_emb:torch.Tensor = self.user_id_emb(user_ids) # (batch, 128)

        # mask the paddings from attention
        mask = (prev_rated_movie_ids != 0).float().unsqueeze(-1) # (batch, prev_rated_seq_len, 1)
        mask = torch.matmul(mask, mask.transpose(-2,-1)).unsqueeze(1).repeat(1,self.num_heads,1,1) # (batch, num_heads, prev_rated_seq_len, prev_rated_seq_len)
        
        rated_movie_emb = self.movie_id_emb(prev_rated_movie_ids)   # (batch, prev_rated_seq_len, 128)
        rated_movie_emb = self.positional_encoding(rated_movie_emb) # (batch, prev_rated_seq_len, 128)
        rated_movie_emb = self.encoder_block(rated_movie_emb, mask) # (batch, prev_rated_seq_len, 128)

        rated_movie_ratings = prev_ratings.unsqueeze(1) # (batch, 1, prev_rated_seq_len)
        # weighted sum of ratings
        rated_movie_emb_weighted = torch.matmul(rated_movie_ratings, rated_movie_emb).squeeze(1) # (batch, 128)

        user_embedding = torch.concat([user_id_emb, rated_movie_emb_weighted], dim=-1) # (batch, 256)
        user_embedding = self.fc(user_embedding) + user_id_emb + rated_movie_emb_weighted # (batch, embedding_size)

        return user_embedding
    
class RecommenderSystem(nn.Module):
    def __init__(
            self, 
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
            embedding_size,
            dropout=0.0
        ) -> None:

        super(RecommenderSystem, self).__init__()

        self.movie_encoder = \
            MovieEncoder\
            (
                movie_id_size, 
                movie_desc_size,
                movie_genres_size,
                movie_year_size, 
                movie_embedding_size, 
                movie_dropout
            )
        
        self.user_encoder = \
            UserEncoder\
            (
                user_id_size, 
                movie_id_size,
                user_embedding_size, 
                user_prev_rated_seq_len, 
                user_num_encoder_layers, 
                user_num_heads, 
                user_dim_ff,
                user_dropout
            )

        self.fc_concat = nn.Linear(user_embedding_size + movie_embedding_size, embedding_size)
        init_weights(self.fc_concat)

        self.fc = nn.Sequential(
            self.fc_concat,
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embedding_size)
        )

        self.cross_features = CrossFeatureLayer(embedding_size, 3, 0.0)

        self.fc_out = nn.Linear(embedding_size, 1)
        init_weights(self.fc_out)

        self.out = nn.Sequential(
            self.fc_out
        )


    def forward(
            self, 
            user_ids:torch.Tensor, # (batch,)
            user_prev_rated_movie_ids:torch.Tensor, # (batch, seq)
            user_prev_ratings:torch.Tensor, # (batch, seq)
            movie_ids:torch.Tensor, # (batch,)
            movie_descriptions:torch.Tensor, # (batch, ntokens)
            movie_genres:torch.Tensor, # (batch, ntokens)
            movie_years:torch.Tensor # (batch,)
        ):
        
        movie_embeddings = \
            self.movie_encoder\
            (
                movie_ids, 
                movie_descriptions, 
                movie_genres, 
                movie_years
            )                                   # (batch, movie_embedding_size)
        
        user_embeddings = \
            self.user_encoder\
                (
                    user_ids, 
                    user_prev_rated_movie_ids, 
                    user_prev_ratings,
                )                               # (batch, user_embedding_size)
        
        emb_concat = torch.concat([movie_embeddings, user_embeddings], dim=-1) # (batch, movie_embedding_size + user_embedding_size)
        
        emb  = self.fc_concat(emb_concat) # (batch, embedding_size)
        emb  = self.cross_features(emb) + emb  # (batch, embedding_size)
        out  = self.out(emb).squeeze(-1)  # (batch,)

        return out
    
