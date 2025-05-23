from threading import local
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML
import pdb
import warnings
import json
import os
import sys
import math
import argparse
from argparse import Namespace
from typing import Optional, Tuple, Literal, List

from torch import dist
warnings.filterwarnings('ignore')

sys.path.append('../')
from scripts.finetune.tune.tune_dataset_details import get_dataset_details

### Run this file from inference folder



models = [
    # 'tv6', 'tv6.1', 'tv6.2', 'tv6.3', 'tv6.4',
    # 'tv7.1',
    # 'tv7.2',
    # 'tv7.3',
    'tv7.4',
    'mlp','xgboost','catboost',"dcnv2",
    # 'ftt',
    # "saint", "tabnet", "autoint",
]



# details of datset
def get_dataset_names(task = 'binclass'):
    # picking from proj_dir/data/finetune-bin 
    if task == 'openml':
        with open('../openml_datasets_sorted.txt') as f:
            dataset_names = [line.strip() for line in f]
        return dataset_names
    elif task == 'binclass':
        path = '../data/finetune-bin/'
        folders = os.listdir(path)
        folders = sorted(folders)
        dataset_names = [x[:-4]  for x in folders if x[-4:] == '.csv']
        return dataset_names
    elif task == 'custom':
        return ['mnist']

def get_db_details_by_rd_data(dataset= "Bank_Personal_Loan_Modelling", reduce_data = 0.8, task = 'binclass'):
    try:
        op = get_dataset_details(reduce_data=reduce_data, model='tv7', dataset=dataset, task=task)
    except:
        op = None
        # print(e)
        return None, None, None, None
    total_features = 0
    if op['dataset'].X_num is not None:
        x_num_shape = [op['dataset'].X_num[phase].shape for phase in ['train', 'val', 'test']]
        for col in range(op['dataset'].X_num['train'].shape[1]):
            total_features += len(set(op['dataset'].X_num['train'][:,col]))
    else:
        x_num_shape = None
    if op['dataset'].X_cat is not None:
        x_cat_shape = [op['dataset'].X_cat[phase].shape for phase in ['train', 'val', 'test']]
        for col in range(op['dataset'].X_cat['train'].shape[1]):
            total_features += len(set(op['dataset'].X_cat['train'][:,col]))
    else:
        x_cat_shape = None
    pos_points = sum([op['dataset'].y[phase].sum() for phase in['train', 'val', 'test']])
    
    return x_num_shape, x_cat_shape, pos_points, total_features

def get_db_details_by_rd(reduce_data = 0.8, task = 'binclass'):
    dataset_names = get_dataset_names(task=task)
    row = []
    cols = {'x_num': [], 'x_cat' : [], 'train_size': [], 'val_size': [], 'test_size': [], 'total_size': [], 'class_ratio': [], 'total_features': []}
    for dataset_name in dataset_names:
        col1, col2, pos_points, total_features = get_db_details_by_rd_data(dataset =dataset_name, reduce_data = reduce_data, task=task)
        if col1 is not None:
            x_num = col1[0][1]
            train_size = col1[0][0]
            val_size = col1[1][0]
            test_size = col1[2][0]

        else:
            x_num = 0

        if col2 is not None:
            x_cat = col2[0][1] 
            train_size = col2[0][0]
            val_size = col2[1][0]
            test_size = col2[2][0]

        else:
            x_cat = 0

        if col1 is None and col2 is None:
            x_num = 0 
            x_cat = 0 
            train_size = 0
            val_size = 0
            test_size = 0
        
        cols['total_features'].append(total_features)
        cols['x_num'].append(x_num) 
        cols['x_cat'].append(x_cat) 
        cols['train_size'].append(train_size)
        cols['val_size'].append(val_size)
        cols['test_size'].append(test_size)
        cols['total_size'].append(train_size + val_size + test_size)
        if cols['total_size'][-1] is not None and cols['total_size'][-1] > 0:
            cols['class_ratio'].append(pos_points/cols['total_size'][-1])
        else:
            cols['class_ratio'].append(None)
            
    return pd.DataFrame(cols, index = dataset_names)


# getting experiment results
def get_result_by_model_db_rd(model, dataset_name, reduce_data, task = 'binclass'):
    result_path = f'../finetune_outputs/{task}-{reduce_data}/{model}-tuned/{dataset_name}/finish.json'
    tuned_path = f'../configs/tuned/{task}-{reduce_data}/{model}/{dataset_name}/cfg.json'
    try:
        with open(result_path, 'r') as f:
            result_json = json.load(f)
    except:
        result_json = None
    try: 
        with open(tuned_path, 'r') as f:
            tuned_json = json.load(f)
    except:
        tuned_json = None
    return result_json, tuned_json

def get_result_by_model_rd(model, reduce_data, dataset_names=None, task='binclass'):
    if dataset_names is None:
        dataset_names = get_dataset_names(task=task)
    
    other_details = {}
    test_score = []
    model_params = []
    additional_params = ['tr_times','test_times','param_total_count','param_total_size', 'model_size']
    for dataset_name in dataset_names:
        other_details[dataset_name] = {}
        result_json, tuned_json = get_result_by_model_db_rd(model=model, dataset_name=dataset_name, reduce_data=reduce_data, task=task)
        if result_json is not None:
            test_score.append(result_json['final_test_score'])
            for k in additional_params:
                other_details[dataset_name][k] = result_json.get(k)
            # other_details['tr_times'].append(result_json.get('tr_times'))
            # other_details['test_times'].append(result_json.get('test_times'))
            # other_details['param_count'].append(result_json.get('param_total_count'))
            # other_details['param_size'].append(result_json.get('param_total_size'))
        else:
            test_score.append(None)
            for k in additional_params:
                other_details[dataset_name][k] = None 
            # other_details['tr_times'].append(None)
            # other_details['test_times'].append(None)
            # other_details['param_count'].append(None)
            # other_details['param_size'].append(None)
        model_params.append(tuned_json)
    
    return test_score, model_params, other_details, dataset_names

def get_result_by_rd(reduce_data, return_model_params = True, return_val_score = True, extra_details = False, task = 'binclass'):
    
    test_score = {}
    val_score = {}
    model_params = {}
    other_details = {}
    for model in models:
        test_score[model], model_params[model], other_details[model], dataset_names = get_result_by_model_rd(model=model, reduce_data=reduce_data, task= task)
        if return_val_score:
            val_score[model] = [x['eval_score'] if x is not None else None for x in model_params[model]]
    
    return_data = [pd.DataFrame(test_score, index=dataset_names)]
    if return_model_params:
        return_data.append(model_params)
    if return_val_score:
        return_data.append(pd.DataFrame(val_score, index=dataset_names))
    if extra_details:
        return_data.append(other_details)
    return return_data
    
# Mergeing exp result and dataset details
def get_result_db_details_by_rd(reduce_data = 0.8, get_time_param_details = False, task='binclass'):
    db_details = get_db_details_by_rd(reduce_data, task = task)
    result_ = get_result_by_rd(reduce_data, extra_details=get_time_param_details, task=task)
    
    test = pd.concat([result_[0], db_details], axis= 1)
    val = pd.concat([result_[2], db_details], axis= 1)
    if get_time_param_details:
        return test, val, result_[-1]
    return test, val


# filtering None and 2.753167e+05NaN values to get filtered dataframe
def filter_none_nan(df_main):
    df = df_main.copy(deep = True)
    # removing models that are not yet run 
    for model in models:
        if df[model].isna().all():
            df.drop(model, axis=1, inplace = True)

    # removing datasets where any model run failed
    filtered_models = [x for x in df.columns if x in models]
    remove_db_rows = []
    for db_name, row in df.iterrows():
        if row[filtered_models].isna().any():
            remove_db_rows.append(db_name)
    for db_name in remove_db_rows:
        df.drop(db_name, axis=0, inplace=True)
    
    return df

def get_rank(df):
    local_models = [x for x in df.columns if x in models]
    db_columns = [x for x in df.columns if x not in models]
    
    rank_dict = {}
    for db_name, row in df.iterrows():
        rank_dict[db_name] = {}
        sorted_val = row[local_models].sort_values(axis = 0)[::-1]
        prev_val = -1
        prev_idx = 0
        for idx, (model, val) in enumerate(sorted_val.items()):
            if val == prev_val:
                rank_dict[db_name][model] = prev_idx
            else:
                rank_dict[db_name][model] = idx + 1
                prev_val = val
                prev_idx = idx + 1
    rank_df = pd.DataFrame(rank_dict).T
    rank_df[db_columns] = df[db_columns]
    return rank_df

def get_percentage_diff(df, type: Literal['diff', 'percent'] = 'diff', precision_digits = 1):
    test_percentage = df.copy(deep = True)
    local_models = [x for x in test_percentage.columns if x in models]
    max_roc = test_percentage[local_models].max(axis = 1)
    min_roc = test_percentage[local_models].min(axis = 1)
    
    for model in local_models:
        if type == 'diff':
            test_percentage[model] = (((max_roc - test_percentage[model])/(max_roc - min_roc)) *10**precision_digits).astype(int)/10**precision_digits
        elif type == 'percent':
            test_percentage[model] = (max_roc - test_percentage[model])*100/max_roc
    return test_percentage
    
def get_hparams(hparams, model, reduce_data= 0.8, task  = 'binclass'):
    distribution = {}
    dataset_names = get_dataset_names(task=task)
    _, model_params, _ = get_result_by_rd(reduce_data, task=task)
    for hparam in hparams: 
        distribution[hparam] = {}
        for idx, db in enumerate(dataset_names):
            local_param = model_params[model][idx]
            if local_param is not None:
                try:
                    distribution[hparam][db] = local_param['model'][hparam]
                except:
                    print(f'No {hparam} in Dataset: {db}')
    return pd.DataFrame(distribution)



def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--get_folder", type=str, default='finetune_outputs')
    parser.add_argument("--root_folder", type=str, default='finetuned_plots')
    parser.add_argument("--task", type=str, choices=['binclass', 'openml'], required=True)
    parser.add_argument("--rel_drop_dbsize", type=str, choices=['score', 'rank'], default='score')
    parser.add_argument("--main_model", type=str, default='tv7.4')
    parser.add_argument("--reduce_data", type=float, default=0.8) # FT-Transformer settings
    args = parser.parse_args()


    thesis_plot = True

    args.root_folder = f'{args.root_folder}/{args.task}-{args.reduce_data}/'
    if args.task == 'openml':
        args.task_name = 'OpenMLCC:18'
    elif args.task == 'binclass':
        args.task_name = 'TabPertNet'
    if not os.path.isdir(args.root_folder):
        os.makedirs(args.root_folder)

    test_comb, val_comb, time_param_details = get_result_db_details_by_rd(
        reduce_data=args.reduce_data,
        get_time_param_details=True,
        task=args.task)

    val_filtered = filter_none_nan(val_comb)
    test_filtered = filter_none_nan(test_comb)
    test_filtered = test_filtered[test_filtered['train_size'] <= 3000]
    print(f'Not None dataset found: {test_filtered.shape[0]}\n')
    # test_filtered.shape

    print('Saving param count and time in csv file...')
    additional_params = ['tr_times','test_times','param_total_count','param_total_size', 'model_size']
    take_avg = [k for k in additional_params if not k.endswith('times')]
    avg_time_param = {}
    count_datasets= []
    for model, val1 in time_param_details.items():
        if model not in test_filtered.columns:
            continue
        avg_time_param[model] = {}
        for ad_param in additional_params:
            avg_time_param[model][ad_param] = 0
        count_db = 0
    
        for dataset_name, val2 in val1.items():
            if dataset_name not in test_filtered.index.to_list():
                continue
            count_db +=1
            for k,v in val2.items():
                if v is None:
                    v = 0
                # Cross checking any None values (will occour if using non common datasets)
                if k not in ['param_total_count', 'param_total_size'] and v == 0:
                    print('------------- Zero found -----------')
                    print(f'{model}, {dataset_name}=> {k}: {v}')
                avg_time_param[model][k] += v
        for k in take_avg:
            avg_time_param[model][k] /=count_db
        count_datasets.append(count_db)
    
            # print(f'{model}')
    # time_param_details
    time_param_df = pd.DataFrame(avg_time_param).T
    # time_param_df
    temp_view = pd.DataFrame()
    temp_view['Training time (s)'] = time_param_df['tr_times'].astype('int')
    temp_view['Test times (s)'] = time_param_df['test_times'].round(3) #*100).astype('int')/100
    temp_view['Param count (1e5)'] = (time_param_df['param_total_count']/100_000).round(1)
    temp_view['Param size (1e5)'] = (time_param_df['param_total_size']/100_000).round(1)
    temp_view['Model size (1e4)'] = (time_param_df['model_size']/10_000).round(1)
    temp_view.to_csv(f'{args.root_folder}/avg_time_and_size.csv')
    print('\n')

    print('Saving model\'s hyperparam distributions...')
    hparams = {'tv7.4': ['n_vectors', 'n_kernel', 'kernel_std', 'use_score', 'use_kernel'],
               'tv7.3': ['n_vectors', 'n_kernel', 'kernel_std', 'use_score', 'use_kernel'],
               'tv7.2': ['n_vectors', 'n_kernel', 'kernel_std', 'use_score', 'use_kernel'],
               'tv7.1': ['n_vectors', 'n_kernel', 'kernel_std', 'use_score', 'use_kernel'],
               'mlp': [],
               'catboost': [],
               'xgboost': [],
               'ftt': [],
               'dcnv2': []
               }
    for model in models:
        if model in hparams:
            if len(hparams[model]) == 0:
                continue
            plot_data = get_hparams(hparams=hparams[model], model=model, reduce_data=args.reduce_data, task=args.task)
            for hparam in hparams[model]:
                sns.displot(data = plot_data[plot_data[hparam].isna() == False], x = hparam)
                plt.savefig(f'{args.root_folder}/{model}-{hparam}-dist.png')
                plt.close()
    # -------
    
    remove_models = ['tv6.4', 'tv6.3', 'tv6.1', 'tv6.2', 'tv6']
    local_cols = [x for x in test_filtered.columns if x not in remove_models]
    test_percentage = get_percentage_diff(test_filtered[local_cols].copy(deep=True), type= 'diff', precision_digits = 2)
    local_models = [x for x in test_percentage.columns if x in models]
    avg_score_drop = ((test_percentage[local_models].mean(axis=0)*100).astype(int)/100).sort_values()
    print('Average drop in scores of model: (max - model) in %')
    print(avg_score_drop)
    print('\n')
    
    print('Saving model\'s score dist plot')
    dist_plot_kind = 'bar' # 'bar' | 'box'
    if thesis_plot:
        temp_plot_df = test_filtered.rename(columns= {args.main_model: 'pond'})
        new_local_model = []
        for model in local_models:
            if model.startswith('tv'):
                new_local_model.append('pond')
            else:
                new_local_model.append(model)
        local_models = new_local_model
    else:
        temp_plot_df = test_filtered

    if dist_plot_kind == 'bar':
        sns.catplot(temp_plot_df[local_models].round(3), kind='bar')
        plt.ylim(bottom=0.65)
    elif dist_plot_kind == 'box':
        sns.catplot(temp_plot_df[local_models].round(3), kind='box')
    # plt.plot(test_filtered[local_models].round(3).mean(axis = 0))
    plt.title(f'Score distribution across {args.task_name}')
    plt.savefig(f'{args.root_folder}/{args.task_name}_score_dist_plot.png', bbox_inches='tight')
    plt.close()
    print('\n')

    print('Plotting score vs dataset max size...')
    plot_quantity = args.rel_drop_dbsize  # 'score' or 'rank'
    markers = {
        # 'tv6': 'g--',
        # 'tv6.1': 'g-',
        # 'tv6.2': 'g-.',
        # 'tv6.3': 'r',
        # 'tv6.4': 'g-.',
        # 'tv7.1': 'r-.',
        'tv7.2': 'r--',
        'tv7.3': 'r-.',
        'tv7.4': 'r.',
        'catboost': 'b',
        'xgboost': 'b--',
        'ftt': 'c',
        'mlp': 'c--',
        'dcnv2': 'c-.',
    }
    markers[args.main_model] = 'r'
    if args.task == 'binclass':
        size_list = [100, 200, 400, 600, 900, 1500, 2000, test_filtered['train_size'].max()]
        # size_list = [600, 1500, 2000, 3000, 5000, 8000, test_filtered['total_size'].max()]
    elif args.task == 'openml':
        size_list = [100, 200, 400, 600, 900, 1500, test_filtered['train_size'].max()] #, 2000, test_filtered['train_size'].max()]

    compare_db_size = {}
    test_complete_dict = {'model': [], 'did': [], 'max_size': [], plot_quantity: []}
    for total_size in size_list:
        test_more_filter = test_filtered[test_filtered['train_size'] <= total_size]
        # test_more_filter = test_more_filter[test_more_filter['total_size'] != 42]
        if plot_quantity == 'score':
            test_percentage = get_percentage_diff(test_more_filter.round(3), type='diff')
        elif plot_quantity == 'rank':
            test_percentage = get_rank(test_more_filter.round(3))
        else:
            raise ValueError(f'Invalid value of {plot_quantity = }')
        local_models = [x for x in test_percentage.columns if x in models]
    
        compare_db_size[total_size] = ((test_percentage[local_models].mean(axis=0)*100).astype(int)/100).sort_values()
        for model in local_models:
            for did in test_percentage.index:
                test_complete_dict['model'].append(model)
                test_complete_dict['did'].append(did)
                test_complete_dict['max_size'].append(total_size)
                test_complete_dict[plot_quantity].append(test_percentage.loc[did, model])
    # test_percentage
    compare_db_size = pd.DataFrame(compare_db_size).T
    test_complete_df = pd.DataFrame(test_complete_dict)
    local_models = [x for x in compare_db_size.columns if x in models]
    for model in local_models:
        # sns.lineplot(data = compare_db_size[model], label = model, style='-')
        if model not in markers.keys():
            continue
    
        plt.plot(compare_db_size[model].index, compare_db_size[model].values, markers[model], label = model )
    plt.xlabel('Datasets with training size less than')
    if plot_quantity == 'score':
        plt.ylabel('score (%): max - model')
        plt.title('2. Models relative drop in score as dataset size increases.')
    else:
        plt.ylabel('avg rank')
        plt.title('2. Models rank as dataset size increases.')
    plt.legend()
    plt.savefig(f'{args.root_folder}/{plot_quantity}_vs_max_dataset_size.png')
    plt.close()
    
    sns.lineplot(data = test_complete_df, x = 'max_size', y = plot_quantity, hue='model', errorbar=('sd', 0.1))
    plt.savefig(f'{args.root_folder}/{plot_quantity}_vs_max_dataset_size_std.png')
    plt.close()
    print('\n')
    

    print('Plotting model\'s score comparision....')
    model_ = args.main_model # 'tv7.4'
    plot_feature = 'train_size' # 'total_size' # 'class_ratio'
    hue_ =  'x_cat' # 'x_num' 'class_ratio'
    compare_with_model_ = 'catboost'  # 'ftt'  'catboost' 
    score_threshold = 0.0 # plot all datapoints
    test_percentage = get_percentage_diff(test_filtered, type= 'diff', precision_digits = 1)
    compare_db_dist= test_percentage.copy(deep=True)
    compare_db_dist[model_] = compare_db_dist[compare_with_model_] - compare_db_dist[model_] 
    idx = (compare_db_dist[model_] > score_threshold) + (compare_db_dist[model_] < -score_threshold)
    sns.scatterplot(data = compare_db_dist.loc[idx], x =plot_feature, y = model_, hue =hue_)
    plt.plot(np.zeros(compare_db_dist[plot_feature].max()), 'g--')
    plt.ylabel(f'{model_}_better_by')
    plt.title(f'ROC of: {model_.capitalize()} - {compare_with_model_.capitalize()}')
    plt.savefig(f'{args.root_folder}/{model_}-vs-{compare_with_model_}-scores.png')
    
    return 0

if __name__ == '__main__':
    main()
