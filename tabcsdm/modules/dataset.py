import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from rdt import HyperTransformer
from rdt.transformers.numerical import GaussianNormalizer
from rdt.transformers.categorical import LabelEncoder
import pdb
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
import os
import json

def get_dtypes(data, name):
    #分类
    dtypes_dict = dict()
    with open(f'dataset/{name}/{name}.json', 'r') as f:
        info = json.load(f) 
    
    column_names = info['column_names'] if info['column_names'] else data.columns.tolist()
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    num_columns = [column_names[i] for i in num_col_idx]
    cat_columns = [column_names[i] for i in cat_col_idx]

    dt = dict()
    trans = dict()
    for col in column_names:
        if col in num_columns:
            dt[col] = 'numerical'
            trans[col] = None
        else:
            dt[col] = 'categorical'
            trans[col] = LabelEncoder()
    
    dtypes_dict['sdtypes'] = dt
    dtypes_dict['transformers'] = trans

    return dtypes_dict

    
class StandardScaler(object):
    def __init__(self):
        self.loc = None
        self.scale = None

    def fit(self, x):
        self.normalizer = sklearn.preprocessing.QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(x.shape[0] // 30, 1000), 10),
            subsample=1000000000,
            random_state=0,
        )        
        return self
    
    def transform(self, x):
        Quantized = self.normalizer.fit_transform(x.to_numpy().reshape(-1, 1))
        imputed = np.nan_to_num(Quantized)
        return imputed
    
    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def fit_invert(self, x, encoded_col):
        self.fit_transform(x)
        return self.normalizer.inverse_transform(encoded_col.to_numpy().reshape(-1,1))

def train_val_test_split(data_df, cat_columns, num_train = 0, num_test = 0):
    total_num = data_df.shape[0]
    idx = np.arange(total_num)
    seed = 1234

    while True:
        np.random.seed(seed)
        np.random.shuffle(idx)

        train_idx = idx[:num_train]
        test_idx = idx[-num_test:]
        train_df = data_df.loc[train_idx]
        test_df = data_df.loc[test_idx]
        flag = 0
        #太多就不判断了
        for i in cat_columns:
            if len(set(train_df[i])) != len(set(data_df[i])):
                flag = 1
                break
        
        if flag == 0:
            break
        else:
            seed += 1
        print(seed)
        print(flag)
    
    return train_df, test_df, seed, train_idx, test_idx    

class DataFrameParser(Dataset):
    def __init__(self, df_path, dtypes_path, train_save_path, test_save_path, name, test_exist = False, target_name = None, new_gen_dtypes = False, ratio = 0.9):
        data = pd.read_csv(df_path)
        self.data = data

        if os.path.exists(dtypes_path) == False:
            dtypes = get_dtypes(data, name)
            new_gen_dtypes = True
        
        self.ht = HyperTransformer()

        if new_gen_dtypes == False:
            with open(dtypes_path, 'r') as file:
                content = file.read()
                config = eval(content)
        else:
            config = dtypes

        self.ht.set_config(config)
        self.ht_data = self.ht.fit_transform(data)
        self.ht_data2 = self.ht_data.copy()
        self.col_info = self.get_col_info(config)

        self.standard_scaler_transform()
        if test_exist == False:
            self.train_num = int(self.ht_data.shape[0] * ratio)
            self.test_num = self.ht_data.shape[0] - self.train_num
            self.train_df, self.test_df, self.seed, self.train_idx, self.test_idx = train_val_test_split(data_df = self.ht_data.copy(), cat_columns=self.cat_column, num_train=self.train_num, num_test=self.test_num)
            data.iloc[self.train_idx,:].to_csv(train_save_path, index=False)
            data.iloc[self.test_idx,:].to_csv(test_save_path, index=False)
        else:
            self.train_df = self.ht_data
        
        self.train_df_classifer = self.train_df

        self.train_cat_df, self.train_num_df, self.label = self.get_train_df(target_name)
        self.train_df = self.train_df.values

    def _getitem_(self, index):
        return self.train_df[index]
    
    def _len_(self):
        return self.train_df.shape[0]
    
    def get_train_df(self, target_name):
        self.cat_col = []
        self.cat_col_info = []
        for col_idx in range(len(self.col_info)):
            if self.col_info[col_idx] > 0 and self.data.columns[col_idx] != target_name:
                self.cat_col.append(col_idx)
                self.cat_col_info.append(self.col_info[col_idx])
            if self.data.columns[col_idx] == target_name:
                self.target = col_idx

        self.num_col = [col for col in range(len(self.col_info)) if col not in self.cat_col and col != self.target]
        self.no_label_col = [col for col in range(len(self.col_info)) if col in self.cat_col or col in self.num_col]

        return self.train_df.iloc[:,self.cat_col].values, self.train_df.iloc[:,self.num_col].values, self.train_df.iloc[:,self.target].values

    def standard_scaler_transform(self):
        encoders = dict()
        for idx, col in enumerate(self.col_info):
            if col == 0:
                column = self.ht_data.columns[idx]
                encoders[column] = StandardScaler().fit(self.ht_data[column])
        self.encoders = encoders
        self.fit_transform()
    
    def fit_transform(self):
        for column, encoder in self.encoders.items():
            self.ht_data[column] = encoder.fit_transform(self.ht_data[column])
    
    def invert_fit(self, encoded_table):
        for column, encoder in self.encoders.items():
            encoded_table[column] = StandardScaler().fit_invert(self.ht_data2.copy()[column], encoded_table[column])
        return encoded_table

    def get_col_info(self, config):
        self.num_numer = 0
        self.cat_column = []
        col_info = []
        for cltype in config['sdtypes'].values():
            if cltype == 'numerical':
                col_info.append(0)
                self.num_numer += 1
            else:
                col_info.append(-1)
                
        for idx, name in enumerate(self.ht_data.columns):
            if col_info[idx] == -1:
                col_info[idx] = self.ht_data[name].nunique()
                self.cat_column.append(name)
        return col_info
    
    def reverse_df(self, df, classifier, target_name):
        self.ht.reset_randomization()
        df = df.cpu()
        df = pd.DataFrame(df.numpy(), columns = self.ht_data.columns)
        for col in df.columns:
            df[col] = df[col].astype(self.train_df_classifer[col].dtypes)

        df[target_name] = classifier.predict(df.iloc[:, self.no_label_col])
        return self.ht.reverse_transform(self.invert_fit(df))

