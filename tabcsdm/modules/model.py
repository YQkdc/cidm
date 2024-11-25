import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

import math
from tqdm import tqdm
import datetime as dt
import pdb

standard_norm = Normal(0, 0.05)
position_norm = Normal(0, 0.01)

class SimpleEmbedding(nn.Module):
    def __init__(self, classes, embedding_dim):
        super(SimpleEmbedding, self).__init__()
        num_embeddings = classes
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data = standard_norm.sample((num_embeddings, embedding_dim))

    def weight(self):
        return self.embedding.weight.data

    def forward(self, idx):
        return self.embedding(idx)

class DynamicLinear(nn.Module):
    def __init__(self, tu):
        super(DynamicLinear, self).__init__()
        self.tu = tu
        self.tl = nn.Parameter(standard_norm.sample([1]))
        self.bias = nn.Parameter(standard_norm.sample([1]))
    
    def forward(self, x, E):
        logits = (torch.matmul(x, E.transpose(1, 0)) + self.bias) / (self.tl * self.tu)
        return logits

class TabMAE(nn.Module):
    def __init__(self, width, depth, heads, dropout, tu, col_info):
        super(TabMAE, self).__init__()
        self.width = width
        self.depth = depth
        self.heads = heads
        self.dropout = dropout
        self.tu = tu
        self.col_info = col_info
        self.lenth = len(col_info)
        self.numer_col = []

        self.Embeddings, self.LinearLayers = nn.ModuleList(), nn.ModuleList()
        for idx, encoder in enumerate(col_info):
            embedding = SimpleEmbedding(encoder, self.width)
            linear = DynamicLinear(self.tu[idx])

            self.Embeddings.append(embedding)
            self.LinearLayers.append(linear)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.width, 
                                                        nhead=self.heads,
                                                        dropout=self.dropout,
                                                        batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.depth)

        self.mask_vec = nn.Parameter(standard_norm.sample((1, self.width)))
        
        self.positional_encoding = nn.Parameter(position_norm.sample((self.lenth, self.width)))
        
        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Total Trainable Parameters in this Model: {pytorch_total_params}')
    
    def embed(self, x, mask):        
        out = self.mask_vec.repeat(x.shape[0], x.shape[1], 1)
        for ft, col in enumerate(self.col_info):
            col_mask = mask[:, ft] == 0
            out[col_mask, ft] = self.Embeddings[ft](x[col_mask, ft].int())
        return out

    def linear(self, x):
        rsl = []
        for ft, col in enumerate(self.col_info):
            rsl.append(self.LinearLayers[ft](x[:, ft], self.Embeddings[ft].weight()))
        return rsl

    def forward(self, x, mask):        
        y = self.embed(x, mask)
        y = y + self.positional_encoding
        y = self.encoder(y)
        y = self.linear(y)
        return y
    


 
