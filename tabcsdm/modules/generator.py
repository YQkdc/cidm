import torch
from modules.model import TabMAE
from tqdm import tqdm
import numpy as np
import pdb
import torch.nn.functional as F

def gen_data(model, tabmae, s0, s1, batch_size, num_col, cat_col, device):
    x = torch.zeros((s0, s1))
    x = x.to(device)
    for rows in tqdm(torch.split(x, batch_size, dim=0), desc='Generating Data'):
        num_rows = rows[:,num_col]
        cat_rows = rows[:,cat_col] 
        mask = torch.ones_like(cat_rows)
        mask = mask.to(device)
        for col in np.random.permutation(range(mask.shape[1])):
            y = tabmae(cat_rows, mask)
            logits = F.softmax(y[col], dim=1)
            predictions = torch.multinomial(logits, num_samples=1)
            predictions = predictions.to(cat_rows[:, col].dtype)
            cat_rows[:, col] = torch.squeeze(predictions)      
            mask[:,col] = 0
        
        for idx, col in enumerate(cat_col):
            rows[:,col]= cat_rows[:,idx]
        
        noise = torch.randn(num_rows.shape, device = device)
        num_rows = model.sample_batch(num_rows.shape[0], num_rows.shape[1], cat_rows, device)
        
        for idx, col in enumerate(num_col):
            rows[:,col]= num_rows[:,idx]

    
    return x