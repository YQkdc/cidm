import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
import tqdm
from tqdm import tqdm
import torch

import pdb


def train_ddpm(dataset, model, lr, epochs, batch_size, weight_decay, name, device, num_col, cat_col):
    
    train_size = dataset.shape[0]
    train_loader = DataLoader(dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              pin_memory=True, 
                              drop_last=True)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = math.ceil(train_size / batch_size) * epochs
    scheduler = CosineAnnealingLR(optimizer, steps)
    
    model.train(True)

    save_path = f'saved_models/{name}/diffusion/ddpm'
    lowest_loss = 10000

    for epoch in range(epochs):
        total_loss = num_loss = 0
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Train Epoch {epoch}") 
                batch = batch.to(device)
                optimizer.zero_grad()
                loss = model(batch, num_col, cat_col)

                loss.backward()
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                num_loss += 1

                tepoch.set_postfix(loss=total_loss / num_loss)
        
        avg_loss = total_loss / num_loss
        
        if avg_loss < lowest_loss:
            lowest_loss = avg_loss
            torch.save(model.state_dict(), save_path)
    
    model.load_state_dict(torch.load(save_path))
    return model