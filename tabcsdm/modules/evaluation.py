import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from catboost import CatBoostClassifier, CatBoostRegressor
#from catboost.metrics import F1

from imblearn.metrics import geometric_mean_score
from modules.generator import gen_data

import pdb

def catboost_regressor(dfp, target_name, depth = 8, iterations=10000):

    real_train_frame = dfp.train_df_classifer
    
    real_train_X = real_train_frame.drop(target_name, axis=1)
    real_train_y = real_train_frame[target_name]

    cat_features = []
    numeric_cols = real_train_X.select_dtypes(include=['int64', 'float64']).columns
    for i, col in enumerate(real_train_X):
        if col not in numeric_cols:
            cat_features.append(i)

    seed = torch.randint(high=100000, size=(1,))
    seed = int(seed)

    reg = CatBoostRegressor(eval_metric='RMSE',
                                    depth = depth,
                                    iterations=iterations,
                                    random_seed=seed)
    reg.fit(
        real_train_X, real_train_y, 
        cat_features=cat_features,
        verbose=False
    )

    return reg



def catboost_classifer(dfp, target_name, depth = 4, iterations=10000):
    real_train_frame = dfp.train_df_classifer
    
    real_train_X = real_train_frame.drop(target_name, axis=1)
    real_train_y = real_train_frame[target_name]

    cat_features = []
    numeric_cols = real_train_X.select_dtypes(include=['int64', 'float64']).columns
    for i, col in enumerate(real_train_X):
        if col not in numeric_cols:
            cat_features.append(i)

    seed = torch.randint(high=100000, size=(1,))
    seed = int(seed)

    classifier = CatBoostClassifier(eval_metric='AUC',
                                    depth = depth,
                                    iterations=iterations,
                                    random_seed=seed)
    
    classifier.fit(
        real_train_X, real_train_y, 
        cat_features=cat_features,
        verbose=False
    )

    return classifier
    


def catboost_trial(train_X, train_y, test_X, test_y, cat_features):

    seed = torch.randint(high=100000, size=(1,))
    seed = int(seed)
    
    classifier = CatBoostClassifier(loss_function='MultiClass',
                                    eval_metric='TotalF1',
                                    iterations=100,
                                    use_best_model=True,
                                    random_seed=seed)
    classifier.fit(
        train_X, train_y, 
        eval_set=(test_X, test_y),
        cat_features=cat_features,
        verbose=False
    )

    predictions = classifier.predict(test_X)
    macrof1 = f1_score(test_y, predictions, average='macro')
    weightedf1 = f1_score(test_y, predictions, average='weighted')
    accuracy = accuracy_score(test_y, predictions)
    macro_gmean = geometric_mean_score(test_y, predictions, average='macro')
    weighted_gmean = geometric_mean_score(test_y, predictions, average='weighted')
    return np.array([accuracy, macrof1, weightedf1, macro_gmean, weighted_gmean])

def compute_catboost_utility(model, tabmae, mae_batch_size, dfp, target_name, num_exp, num_trials, device):
    real_train_frame = dfp.data.iloc[dfp.train_idx,:]
    real_test_frame = dfp.data.iloc[dfp.test_idx,:]

    real_train_y, real_test_y = real_train_frame[target_name], real_test_frame[target_name]
    real_train_X, real_test_X = real_train_frame.drop(target_name, axis=1), real_test_frame.drop(target_name, axis=1)
    
    cat_features = []
    numeric_cols = real_train_X.select_dtypes(include=['int64', 'float64']).columns
    for i, col in enumerate(real_train_X):
        if col not in numeric_cols:
            cat_features.append(i)

    real_results = catboost_trial(real_train_X, real_train_y, real_test_X, real_test_y, cat_features)

    print(f'Performance of the Classifier Trained on Real Data: {real_results}.')
    
    avg_results = []
    for exp in range(num_exp):

        synthetics = gen_data(model, tabmae, dfp.train_df.shape[0], dfp.train_df.shape[1], mae_batch_size, dfp.num_col, dfp.cat_col, device)
        synthetics = dfp.reverse_df(synthetics)
        
        syn_y = synthetics[target_name]
        syn_X = synthetics.drop(target_name, axis=1)
    
        trial_results = []
        for trial in range(num_trials):        
            fake_results = catboost_trial(syn_X, syn_y, real_test_X, real_test_y, cat_features)
            #trial_results.append(real_results - fake_results)
            trial_results.append(fake_results)
            
        trial_results = np.stack(trial_results)
        trial_results = np.mean(trial_results, axis=0)
        
        avg_results.append(trial_results)
    
    avg_results = np.stack(avg_results)
    means = np.mean(avg_results, axis=0)
    stds = np.std(avg_results, axis=0)
    return means, stds, real_results

