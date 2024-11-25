from modules.dataset import DataFrameParser
from modules.model import TabMAE
from modules.train import fit
import numpy as np
import pandas as pd
import torch
import random
import pdb
from tqdm import tqdm
from modules.train import CondDiff
from modules.train_ddpm import train_ddpm
from modules.generator import gen_data
from argparse import ArgumentParser
from modules.evaluation import compute_catboost_utility, catboost_classifer, catboost_regressor

def main():
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, default='credit')
    parser.add_argument('--test_exist', type=bool, default=False)
    parser.add_argument('--ratio', type=float, default=0.9)
    #config for tabmae
    parser.add_argument('--mae_width', type=int, default=256)
    parser.add_argument('--mae_depth', type=int, default=8)
    parser.add_argument('--mae_heads', type=int, default=8)
    parser.add_argument('--mae_dropout', type=float, default=0.005)
    parser.add_argument('--mae_lr', type=float, default=5e-3)
    parser.add_argument('--mae_epochs', type=int, default=100)
    parser.add_argument('--mae_batch_size', type=int, default=256)
    parser.add_argument('--mae_weight_decay', type=float, default=0.001)
    #config for conditional ddpm
    parser.add_argument('--cd_width', type=int, default=256)
    parser.add_argument('--cd_dropout', type=float, default=0.001)
    parser.add_argument('--cd_layers', type=list, default=[256,256])
    parser.add_argument('--cd_lr', type=float, default=5e-4)
    parser.add_argument('--cd_epochs', type=int, default=100)
    parser.add_argument('--cd_batch_size', type=int, default=256)
    parser.add_argument('--cd_weight_decay', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--classifier', type=int, default=1)
    #config for evaluation
    parser.add_argument('--num_exp', type=int, default=5)
    parser.add_argument('--trials_per_exp', type=int, default=5)
    parser.add_argument('--target_name', type=str, default='Term')

    args = parser.parse_args()
    
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    df_path = f'dataset/{args.name}/{args.name}'
    dtypes_path = f'dataset/{args.name}/dtypes.txt'
    train_save_path = f'dataset/{args.name}/train.csv'
    test_save_path = f'dataset/{args.name}/test.csv'
    syn_save_path = f'dataset/{args.name}/syn.csv'
    mae_save_path = f'saved_models/{args.name}/mae/mae'
    classifier_save_path = f'saved_models/{args.name}/classifer/classifer'

    dfp = DataFrameParser(df_path, dtypes_path, train_save_path, test_save_path, args.name, args.test_exist, args.target_name, ratio=args.ratio)

    tu = [1 for i in range(dfp.train_cat_df.shape[1])]
    
    tabmae = TabMAE(width=args.mae_width, 
              depth=args.mae_depth, 
              heads=args.mae_heads, 
              dropout=args.mae_dropout, 
              tu=tu,
              col_info=dfp.cat_col_info
              )

    tabmae = tabmae.to(args.device)
    
    assert dfp.train_cat_df.shape[1] == len(dfp.cat_col_info), "Wrong."
    
    if dfp.train_cat_df.shape[1] > 0:

        tabmae= fit(model=tabmae,
            dataset=dfp.train_cat_df, 
            train_size=dfp.train_cat_df.shape[0],
            col_info=dfp.cat_col_info,
            lr=args.mae_lr, 
            epochs=args.mae_epochs, 
            batch_size=args.mae_batch_size, 
            weight_decay=args.mae_weight_decay,
            save_path=mae_save_path,
            device=args.device)
    
    model = CondDiff(
            input_dim=len(dfp.num_col),
            hidden_dim=args.cd_width,
            cat_cond = dfp.cat_col_info,
            device=args.device,
            dropout=args.cd_dropout,
            d_layers=args.cd_layers
            )

    model = model.to(args.device)
    
    model = train_ddpm(dfp.train_df, model, args.cd_lr, args.cd_epochs, args.cd_batch_size, args.cd_weight_decay, args.name, args.device, dfp.num_col,  dfp.cat_col)
    

    print('Starting Evaluation!')
    model_save_path = f'saved_models/{args.name}/diffusion/ddpm'
    model.load_state_dict(torch.load(model_save_path))
    if dfp.train_cat_df.shape[1] > 0:
        tabmae.load_state_dict(torch.load(mae_save_path))
    model.eval()
    tabmae.eval()
    
    gdata = gen_data(model, tabmae, dfp.train_df.shape[0], dfp.train_df.shape[1], args.mae_batch_size, dfp.num_col, dfp.cat_col, args.device)

    if args.classifier == 1:
        pred = catboost_classifer(dfp, args.target_name)
        pred.save_model(classifier_save_path)
    
    else:
        pred = catboost_regressor(dfp, args.target_name)
        pred.save_model(classifier_save_path)
    
    gdata = dfp.reverse_df(gdata, pred, args.target_name)
    gdata.to_csv(syn_save_path, columns = dfp.ht_data.columns, index = 0)
    '''
    
    means, stds, real_results = compute_catboost_utility(model=model,
                                       tabmae=tabmae,
                                       mae_batch_size=args.mae_batch_size,
                                       dfp=dfp,
                                       target_name=args.target_name,
                                       num_trials=args.trials_per_exp, 
                                       num_exp=args.num_exp,
                                       device=args.device)
    
    results = {"mean_accuracy": means[0],
           "mean_macroF1":means[1],
           "mean_weightedF1":means[2],
           "mean_macroGM":means[3],
           "mean_weightedGM":means[4],
           "std_accuracy": stds[0],
           "std_macroF1":stds[1],
           "std_weightedF1":stds[2],
           "std_macroGM":stds[3],
           "std_weightedGM":stds[4]}
    print(results)

    real_rsl = {
           "real_accuracy": real_results[0],
           "real_macroF1":real_results[1],
           "real_weightedF1":real_results[2],
           "real_macroGM":real_results[3],
           "real_weightedGM":real_results[4],
    }
    print(real_rsl)
    '''

if __name__ == '__main__':    
    main()