import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from modules.tabddpm import MLPDiffusion, loss_fn, Euler_Maruyama_sampling
import pdb
import tqdm
import math

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

g = torch.Generator()
g.manual_seed(42)

import torch
import torch.nn as nn

class CondDiff(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, cat_cond: list, device, dropout: float = 0.0, d_layers:list=[256, 256]):
        super(CondDiff, self).__init__()
        self.hidden_dim = hidden_dim
        rtdl_params={
        'd_in': input_dim,
        'd_layers': d_layers,
        'dropout': dropout,
        'd_out': input_dim,
        }
        self.mlp = MLPDiffusion(input_dim, rtdl_params, hidden_dim)
        self.cat_embedding = nn.ModuleList()
        self.get_cat_embed(cat_cond, hidden_dim)
        self.device = device
    
    def get_cat_embed(self, cat_cond, hidden_dim):
        for cat in cat_cond:
            self.cat_embedding.append(nn.Embedding(cat, hidden_dim))
    
    def forward(self, x, num_col, cat_col):
        cond = x[:,cat_col]
        x = x[:,num_col]
        loss_values = loss_fn(self.mlp, x, 100, cond, self.cat_embedding, self.device)
        loss = torch.mean(loss_values)
        return loss

    def sample_batch(self, N, P, cond, device):
        x = Euler_Maruyama_sampling(self.mlp, 100, N, P, device, cond, self.cat_embedding)
        return x

    def loss(self, x, num_col, cat_col):
        cond = x[:,cat_col]
        x = x[:,num_col]
        noise = torch.randn(x.shape)
        timestep = torch.randint(0, self.diffusion.num_timesteps, (x.shape[0],))
        timestep, noise = timestep.to(self.device), noise.to(self.device)
        x_t = self.diffusion.q_sample(x, timestep, noise)
        noise_pred = self.mlp(x_t, timestep, cond , self.cat_embedding)
        loss = ((noise_pred - noise) ** 2).mean()
        return loss
    
    def sample(self, z, noise):
        x = noise
        for t in list(range(self.diffusion.num_timesteps))[::-1]:
            x, _ = self.diffusion.p_sample(self.mlp, x, self.cat_embedding, t, z)
        return x


def gen_mask(rows):
    mask = torch.rand(rows.shape).round().int()
    return mask

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def get_numer(col_info):
    tmp = []
    for idx, col in enumerate(col_info):
        if col == 0:
            tmp.append(idx)
    return tmp

def train(dataloader, model, col_info, device):
    total_loss = total_correct = item_count = 0
    with tqdm.tqdm(dataloader, unit="batch") as tepoch:
        for batch in tepoch:
            tepoch.set_description(f"Train Epoch {epoch}") 
            rows = batch
            mask = gen_mask(rows)

            rows, mask =  rows.to(device), mask.to(device)
            optimizer.zero_grad()
            y = model(rows, mask)
            loss = 0

            for ft, y_ft in enumerate(y):
                m = (mask[:, ft] == 1) & (rows[:, ft] != -1)
                truth = rows[m, ft]
                pred = y_ft[m]

                if (len(truth) > 0):
                    loss += criterion(pred, truth.long())
                    total_correct += sum(pred.argmax(dim=1) == truth).item()
                    item_count += pred.shape[0]

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            
            tepoch.set_postfix(loss=total_loss / item_count)
        
    return total_loss / item_count
    
def fit(model,
        dataset,  
        train_size,
        col_info,
        lr, 
        epochs, 
        batch_size, 
        weight_decay,
        save_path,
        device
        ):
    
    global optimizer, scheduler, criterion, epoch
    
    train_loader = DataLoader(dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              drop_last=True)
    
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = math.ceil(train_size / batch_size) * epochs
    scheduler = CosineAnnealingLR(optimizer, steps)
    
    model = model.to(device)

    model.train(True)
    
    lowest_loss = 10000
    
    for epoch in range(epochs):
        t_loss = train(train_loader, model, col_info, device)
        
        if t_loss < lowest_loss:
            lowest_loss = t_loss
            torch.save(model.state_dict(), save_path)


    model.load_state_dict(torch.load(save_path))
    return model