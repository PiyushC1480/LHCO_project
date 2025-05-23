import os
import pdb
import sys
import gc

from transformers.models.perceiver.modeling_perceiver import PreprocessorType
sys.path.append(os.getcwd()) # to correctly import bin & lib
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import optuna
import warnings
warnings.filterwarnings("ignore")

from lib import DataConfig, data_preproc, prepare_tensors, make_optimizer, calculate_metrics, get_openml_db, get_custom_db
from bin import MLP, AutoInt, DCNv2, SAINT, TV6, TV7

def get_training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, default='configs/tuned')
    parser.add_argument("--dataset", type=str, default='train_1748_Sales_DataSet_of_SuperMarket')
    parser.add_argument("--task", type=str, choices=['custom', 'binclass', 'openml', 'regression', 'multiclass'], required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--early_stop", type=int, default=16) # FT-Transformer settings
    parser.add_argument("--reduce_data", type=float, default=0.8) # FT-Transformer settings
    args = parser.parse_args()

    args.output = f'{args.output}/{args.task}-{args.reduce_data}/{args.model}/{args.dataset}'
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    
    return args

def save_result(
    args, 
    best_ev, final_test, 
    tr_losses,
    ev_metrics, 
    test_metrics, 
    suffix
):
    saved_results = {
        'args': vars(args),
        'device': torch.cuda.get_device_name(),
        'best_eval_score': best_ev,
        'final_test_score': final_test,
        'ev_metric': ev_metrics,
        'test_metric': test_metrics,
        'tr_loss': tr_losses,
    }
    with open(Path(args.output) / f'{suffix}.json', 'w') as f:
        json.dump(saved_results, f, indent=4)

def seed_everything(seed=42):
    '''
    Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.
    '''
    random.seed(seed)
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


"""args"""
device = torch.device('cuda')
args = get_training_args()
seed_everything(seed=42)

""" prepare Datasets and Dataloaders """
if args.task == 'binclass':
    from lib import BIN_CHECKPOINT as CHECKPOINT_DIR
    from lib import FINETUNE_BIN_DATA as FINETUNE_DATA
elif args.task == 'regression':
    from lib import BIN_CHECKPOINT as CHECKPOINT_DIR
    from lib import FINETUNE_REG_DATA as FINETUNE_DATA
elif args.task == 'multiclass':
    from lib import BIN_CHECKPOINT as CHECKPOINT_DIR
    from lib import FINETUNE_MUL_DATA as FINETUNE_DATA
elif args.task == 'openml':
    from lib import OPENML_CHECKPOINT as CHECKPOINT_DIR
    from lib import FINETUNE_OPENML_DATA as FINETUNE_DATA
elif args.task == 'custom':
    from lib import CUSTOM_CHECKPOINT as CHECKPOINT_DIR
    from lib import FINETUNE_CUSTOM_DATA as FINETUNE_DATA

# if args.model == 'tv6':
#     preproc_type = 'xgboost'
# else:
#     preproc_type = 'ftt'
preproc_type = 'ftt'
if args.task == 'openml':
    dataset = get_openml_db(
        CHECKPOINT_DIR,
        FINETUNE_DATA,
        dataset_id=args.dataset,
        reduce_data= args.reduce_data,
        preproc_type = preproc_type)
elif args.task == 'binclass':
    data_config = DataConfig.from_pretrained(
        CHECKPOINT_DIR, data_dir=FINETUNE_DATA,
        batch_size=64, train_ratio=0.8,
        preproc_type=preproc_type, reduce_data = args.reduce_data, pre_train=False)
    dataset = data_preproc(args.dataset, data_config, no_str=True, tt=args.task)
elif args.task == 'custom':
    dataset = get_custom_db(
        CHECKPOINT_DIR,
        dataset_name= args.dataset,
        reduce_data= args.reduce_data,
        preproc_type = preproc_type, seed = 42)

if args.model == 'saint' and dataset.X_num is None: # SAINT original implementation requires at least one numerical features
    new_Xnum = {k: v[:, :1].astype(np.float32) for k, v in dataset.X_cat.items()} # treat the first categorical one as numerical
    new_Xcat = {k: v[:, 1:] for k, v in dataset.X_cat.items()}
    from dataclasses import replace
    dataset = replace(dataset, X_num=new_Xnum, X_cat=new_Xcat)

d_out = dataset.n_classes or 1
X_num, X_cat, ys = prepare_tensors(dataset, device=device)

batch_size = args.batch_size
val_batch_size = 64


# data loaders
check_list = {}
for x in ['X_num', 'X_cat']:
    check_list[x] = False if eval(x) is None else True

data_list = [x for x in [X_num, X_cat, ys] if x is not None]
train_dataset = TensorDataset(*(d['train'] for d in data_list))
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
)
val_dataset = TensorDataset(*(d['val'] for d in data_list))
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    shuffle=False,
)
test_dataset = TensorDataset(*(d['test'] for d in data_list))
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=val_batch_size,
    shuffle=False,
)
dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

""" Hyper-parameter Spaces """
model_param_spaces = {
    "tv7": { # currently same as 7.1
        "n_vectors": (10, 500, 'int'),
        "n_kernel": (10, 300, 'int'),
        "kernel_std": (0.5, 1, 'float'),
        "use_kernel": (['rbf', 'lin'], 'categorical'),
        "use_score": (['lin', 'rbf'], 'categorical'),
        "normalize_proj_diff": 'max-detached', # 'max/n_kernel', 'max-detach'
        "detached_vec_norm": True,
        "detached_pivots": False,
        "use_svm_score": True,
        # "diff_kernel": False,
        "train_gamma": False,
        "use_kernel_weight": True,
        "train_std": False,
        # "reg_type": 'lin',
        "reg_wt": ( 0.001, 0.1, 'loguniform'),
        "combine_type": 'embd',  # Use default when doing ohe, else use 'embd'
        # "loss_type": 'svml'
    },
    "tv7.1": {  ## using hinge loss, 
        "n_vectors": (10, 1000, 'int'),
        "n_kernel": (5, 6, 'later'),
        "kernel_std": (0.1, 1.0, 'float'),
        "use_kernel": (['rbf', 'lin'], 'categorical'),
        # "use_score": 'lin',
        "use_score": (['lin', 'rbf'], 'categorical'),
        "normalize_proj_diff": 'max-detached', # 'max/n_kernel', 'max-detach'
        "detached_vec_norm": True,
        "detached_pivots": False,
        "use_svm_score": True,
        # "diff_kernel": False,
        "train_gamma": False,
        "use_kernel_weight": True,
        "train_std": False,
        # "reg_type": 'lin',
        "reg_wt": 0.0,
        "combine_type": 'embd',  # TODO: use default or embd?
        # "loss_type": 'svml'
    },
    "tv7.2": {  ## Using krenel_std from 05. ot 1, ## Currently changed to combine_type to default
        # using only linear kernel and lin + rbg weights, so like 7.4 but dataset is normalized
        "n_vectors": (10, 1000, 'int'),
        "n_kernel": (5, 6, 'later'),
        "kernel_std": (0.1, 1.0, 'float'),
        "use_kernel": 'lin',
        # "use_kernel": (['rbf', 'lin'], 'categorical'),
        "use_score": (['lin', 'rbf'], 'categorical'),
        "normalize_proj_diff": 'max-detached', # 'max/n_kernel', 'max-detach'
        "detached_vec_norm": True,
        "detached_pivots": False,
        "use_svm_score": True,
        # "diff_kernel": False,
        "train_gamma": False,
        "use_kernel_weight": True,
        "train_std": False,
        # "reg_type": 'lin',
        "reg_wt": ( 0.001, 0.1, 'loguniform'),
        "combine_type": 'embd',  # Use default when doing ohe, else use 'embd'
        # "loss_type": 'svml'
    },
    "tv7.3": {  ## default = odrinal , (Not using ohe = OHE)
        "n_vectors": (10, 800, 'int'),
        "n_kernel": (5, 6, 'later'),
        "kernel_std": (0.1, 0.5, 'float'),
        "use_kernel": 'lin',
        # "use_kernel": (['rbf', 'lin'], 'categorical'),
        "use_score": (['lin', 'rbf'], 'categorical'),
        "normalize_proj_diff": 'max-detached', # 'max/n_kernel', 'max-detach'
        "detached_vec_norm": True,
        "detached_pivots": False,
        "use_svm_score": True,
        # "diff_kernel": False,
        "train_gamma": False,
        "use_kernel_weight": True,
        "train_std": False,
        # "reg_type": 'lin',
        "reg_wt": ( 0.001, 0.1, 'loguniform'),
        "combine_type": 'default',  # Use default when doing ohe, else use 'embd'
        # "loss_type": 'svml'
    },
    "tv7.4": {  # NOTE: 600-> 1000 for openml ## Removed 0.5 + hinge loss + embd + small kernel_std
        "n_vectors": (5, 800, 'int'),
        "n_kernel": (5,6,'later'),
        "kernel_std": (0.1, 0.5, 'float'),
        # "use_kernel": (['rbf', 'lin'], 'categorical'),
        "use_kernel": 'lin',
        "use_score": (['lin', 'rbf'], 'categorical'),
        "normalize_proj_diff": 'max-detached', # 'max/n_kernel', 'max-detach'
        "detached_vec_norm": True,
        "detached_pivots": False,
        "use_svm_score": True,
        # "diff_kernel": False,
        "train_gamma": False,
        "use_kernel_weight": True,
        "train_std": False,
        # "reg_type": 'lin',
        "reg_wt": ( 0.001, 0.1, 'loguniform'),
        "combine_type": 'embd',  # Use default when doing ohe, else use 'embd'
        # "loss_type": 'svml'
    },
    "tv6": {
        "n_vectors": (10, 500, 'int'),
        "n_kernel": (10, 300, 'int'),
        "kernel_std": (0.5, 1, 'float'),
        "use_kernel": (['rbf', 'lin'], 'categorical'),
        "use_score": (['lin', 'rbf'], 'categorical'),
        "normalize_proj_diff": 'max-detached', # 'max/n_kernel', 'max-detach'
        "detached_vec_norm": True,
        "detached_pivots": False,
        "use_svm_score": True,
        # "diff_kernel": False,
        "train_gamma": False,
        "use_kernel_weight": True,
        "train_std": False,
        "reg_type": 'lin',
        "reg_wt": ( 0.001, 0.1, 'loguniform'),
        "combine_type": 'embd',  # Use default when doing ohe, else use 'embd'
        # "loss_type": 'svml'
    },
    "tv6.1": {  ## Removed 0.5 from output and using hinge loss, 
        "n_vectors": (10, 500, 'int'),
        "n_kernel": (10, 300, 'int'),
        "kernel_std": (0.5, 1, 'float'),
        "use_kernel": (['rbf', 'lin'], 'categorical'),
        "use_score": (['lin', 'rbf'], 'categorical'),
        "normalize_proj_diff": 'max-detached', # 'max/n_kernel', 'max-detach'
        "detached_vec_norm": True,
        "detached_pivots": False,
        "use_svm_score": True,
        # "diff_kernel": False,
        "train_gamma": False,
        "use_kernel_weight": True,
        "train_std": False,
        "reg_type": 'lin',
        "reg_wt": ( 0.001, 0.1, 'loguniform'),
        "combine_type": 'embd',  # Use default when doing ohe, else use 'embd'
        # "loss_type": 'svml'
    },
    "tv6.2": {  ## removed 0.5 from output, no hinge loss, small n_vectors, n_kernel
        "n_vectors": (10, 200, 'int'),
        "n_kernel": (10, 200, 'int'),
        "kernel_std": (0.5, 1, 'float'),
        "use_kernel": (['rbf', 'lin'], 'categorical'),
        "use_score": (['lin', 'rbf'], 'categorical'),
        "normalize_proj_diff": 'max-detached', # 'max/n_kernel', 'max-detach'
        "detached_vec_norm": True,
        "detached_pivots": False,
        "use_svm_score": True,
        # "diff_kernel": False,
        "train_gamma": False,
        "use_kernel_weight": True,
        "train_std": False,
        "reg_type": 'lin',
        "reg_wt": ( 0.001, 0.1, 'loguniform'),
        "combine_type": 'embd',  # Use default when doing ohe, else use 'embd'
        # "loss_type": 'svml'
    },
    "tv6.3": {  ## Removed 0.5 from output and using hinge loss, OHE
        "n_vectors": (10, 500, 'int'),
        "n_kernel": (10, 300, 'int'),
        "kernel_std": (0.5, 1, 'float'),
        "use_kernel": (['rbf', 'lin'], 'categorical'),
        "use_score": (['lin', 'rbf'], 'categorical'),
        "normalize_proj_diff": 'max-detached', # 'max/n_kernel', 'max-detach'
        "detached_vec_norm": True,
        "detached_pivots": False,
        "use_svm_score": True,
        # "diff_kernel": False,
        "train_gamma": False,
        "use_kernel_weight": True,
        "train_std": False,
        "reg_type": 'lin',
        "reg_wt": ( 0.001, 0.1, 'loguniform'),
        "combine_type": 'ohe',  # Use default when doing ohe, else use 'embd'
        # "loss_type": 'svml'
    },
    "tv6.4": {  ## Removed 0.5 + hinge loss + embd + small kernel_std
        "n_vectors": (5, 700, 'int'),
        "n_kernel": (5, 400, 'int'),
        "kernel_std": (0.1, 0.5, 'float'),
        "use_kernel": (['rbf', 'lin'], 'categorical'),
        "use_score": (['lin', 'rbf'], 'categorical'),
        "normalize_proj_diff": 'max-detached', # 'max/n_kernel', 'max-detach'
        "detached_vec_norm": True,
        "detached_pivots": False,
        "use_svm_score": True,
        # "diff_kernel": False,
        "train_gamma": False,
        "use_kernel_weight": True,
        "train_std": False,
        "reg_type": 'lin',
        "reg_wt": ( 0.001, 0.1, 'loguniform'),
        "combine_type": 'embd',  # Use default when doing ohe, else use 'embd'
        # "loss_type": 'svml'
    },
    "mlp": {
        "n_layers": (1,16,'int'),
        "first_dim": (1,1024, 'int'),
        "mid_dim": (1,1024, 'int'),
        "last_dim": (1,1024, 'int'),
        "dropout": (0, 0.5, 'uniform'),
    },
    'autoint': {
        'activation': 'relu',
        'initialization': 'kaiming',
        'n_heads': 2,
        'prenormalization': False,
        'attention_dropout': (0, 0.5, 'uniform'),
        'd_token': (8, 64, 2, 'int'),
        'n_layers': (1, 6, 'int'),
        'residual_dropout': (0, 0.5, 'uniform'),
    },
    "dcnv2": {
        "cross_dropout": (0, 0.5, 'uniform'),
        "d": (64,512,'int'),
        "hidden_dropout": (0, 0.5, 'uniform'),
        "n_cross_layers": (1,8,'int'),
        "n_hidden_layers": (1,8,'int'),
        "stacked": False
    },
    "saint": {
        # default configs
    },
}
tv_kernel_parameters = {
    'rbf': {
        "gamma": (['auto', 0.01, 0.1, 1.0], 'categorical')
    },
    'lin': {}
}
d_embedding_dicts = {
    'dcnv2': (64,128,'int'),
    'mlp': (64,128,'int'),
    "node": 256,
    "tv6": (64, 128, 'int'),
    "tv6.1": (64, 128, 'int'),
    "tv6.2": (64, 128, 'int'),
    "tv6.3": (64, 128, 'int'),
    "tv6.4": (64, 128, 'int'),
    "tv7": (64, 128, 'int'),
    "tv7.1": (64, 128, 'int'),
    "tv7.2": (64, 128, 'int'),
    "tv7.3": 64,
    "tv7.4": (64, 128, 'int'), # reduced from 512 to fiit some datasets in memory
}
if args.model in d_embedding_dicts and dataset.X_cat is not None:
    model_param_spaces[args.model]['d_embedding'] = d_embedding_dicts[args.model]
training_param_spaces = {
    'tv6': {
        'lr': (1e-5, 1e-3, 'loguniform'),
        'weight_decay': 0.0,
        'optimizer': 'adamw'
    },
    'tv6.1': {
        'lr': (1e-5, 1e-3, 'loguniform'),
        'weight_decay': 0.0,
        'optimizer': 'adamw'
    },
    'tv6.2': {
        'lr': (1e-5, 1e-3, 'loguniform'),
        'weight_decay': 0.0,
        'optimizer': 'adamw'
    },
    'tv6.3': {
        'lr': (1e-5, 1e-3, 'loguniform'),
        'weight_decay': 0.0,
        'optimizer': 'adamw'
    },
    'tv6.4': {
        'lr': (1e-5, 1e-3, 'loguniform'),
        'weight_decay': 0.0,
        'optimizer': 'adamw'
    },
    'tv7': {
        'lr': (1e-5, 1e-3, 'loguniform'),
        'weight_decay': 0.0,
        'optimizer': 'adamw'
    },
    'tv7.1': {
        'lr': (1e-5, 1e-3, 'loguniform'),
        'weight_decay': 0.0,
        'optimizer': 'adamw'
    },
    'tv7.2': {
        'lr': (1e-5, 1e-3, 'loguniform'),
        'weight_decay': 0.0,
        'optimizer': 'adamw'
    },
    'tv7.3': {
        'lr': (1e-5, 1e-3, 'loguniform'),
        'weight_decay': 0.0,
        'optimizer': 'adamw'
    },
    'tv7.4': {
        'lr': (1e-5, 1e-3, 'loguniform'),
        'weight_decay': 0.0,
        'optimizer': 'adamw'
    },
    'mlp': {
        'lr': (1e-5, 1e-2, 'loguniform'),
        'weight_decay': (1e-6, 1e-3, 'loguniform'),
        'optimizer': 'adamw'
    },
    'autoint': {
        'lr': (1e-5, 1e-3, 'loguniform'),
        'weight_decay': (1e-6, 1e-3, 'loguniform'),
        'optimizer': 'adamw'
    },
    'dcnv2': {
        'lr': (1e-5, 1e-2, 'loguniform'),
        'weight_decay': (1e-6, 1e-3, 'loguniform'),
        'optimizer': 'adamw'
    },
    'saint': {
        'lr': (1e-5, 1e-2, 'loguniform'),
        'weight_decay': (1e-6, 1e-3, 'loguniform'),
        'optimizer': 'adamw'
    },
}

""" Metric Settings """
metric_key = {
    'regression': 'rmse', 
    'binclass': 'roc_auc', 
    'multiclass': 'accuracy'
}[dataset.task_type.value]
scale = 1 if not dataset.is_regression else -1

def get_model_training_params(trial):
    model_args = model_param_spaces[args.model]
    training_args = {
        'batch_size': batch_size,
        'eval_batch_size': val_batch_size,
        **training_param_spaces[args.model],
    }
    model_params = {}
    training_params = {}
    for param, value in model_args.items():
        if isinstance(value, tuple):
            suggest_type = value[-1]
            if suggest_type == 'later':
                continue
            if suggest_type != 'categorical':
                model_params[param] = eval(f'trial.suggest_{suggest_type}')(param, *value[:-1])
            else:
                model_params[param] = trial.suggest_categorical(param, choices=value[0])
        else:
            model_params[param] = value
        if param == 'use_kernel' and args.model.startswith('tv7'):  # only for TV7
            if model_params[param] == 'rbf':
                temp_val = tv_kernel_parameters['rbf']["gamma"][0]
                model_params['gamma'] = trial.suggest_categorical("gamma", choices = temp_val)
    for param, value in training_args.items():
        if isinstance(value, tuple):
            suggest_type = value[-1]
            if suggest_type != 'categorical':
                training_params[param] = eval(f'trial.suggest_{suggest_type}')(param, *value[:-1])
            else:
                training_params[param] = trial.suggest_categorical(param, choices=value[0])
        else:
            training_params[param] = value
    return model_params, training_params

def process_mlp_params(params):
    d_layers = []
    for i in range(params['n_layers']):
        if i == 0:
            d_layers.append(params['first_dim'])
        elif i == params['n_layers'] - 1 and params['n_layers'] > 1:
            d_layers.append(params['last_dim'])
        else:
            d_layers.append(params['mid_dim'])
    params['d_layers'] = d_layers
    del params['n_layers'], params['first_dim'], params['mid_dim'], params['last_dim']
    return params

def objective(trial):
    cfg_model, cfg_training = get_model_training_params(trial)
    # NOTE: changed from train to all
    cats = dataset.get_category_sizes('all')
    if len(cats) == 0:
        cats = None
    """set default"""
    if args.model.startswith('tv6'):
        cfg_model.setdefault('d_embedding', None)
        model = TV6(
            d_in=dataset.n_num_features,
            categories=cats,
            **cfg_model
        )
        model.init_phase2(dataset, n_kernel=cfg_model['n_kernel'])  # For setting kernel points
        model.to(device)
        # model.parameters.astype(torch.float32)
    elif args.model.startswith('tv7'):
        cfg_model.setdefault('d_embedding', None)
        max_n_kernel = dataset.y['train'].shape[0]
        max_n_kernel = np.min([max_n_kernel, 1000]) # added later for openml
        cfg_model['n_kernel'] = trial.suggest_int('n_kernel', 5, max_n_kernel)
        model = TV7(
            d_in=dataset.n_num_features,
            categories=cats,
            **cfg_model
        )
        model.init_phase2(dataset, n_kernel=cfg_model['n_kernel'])  # For setting kernel points
        model.to(device)
    elif args.model == 'mlp':
        cfg_model.setdefault('d_embedding', None)
        cfg_model = process_mlp_params(cfg_model)
        model = MLP(
            d_in=dataset.n_num_features,
            categories=cats,
            d_out=d_out,
            **cfg_model
        ).to(device)
    elif args.model == 'autoint':
        cfg_model.setdefault('kv_compression', None)
        cfg_model.setdefault('kv_compression_sharing', None)
        model = AutoInt(
            d_numerical=dataset.n_num_features,
            categories=cats,
            d_out=d_out,
            **cfg_model
        ).to(device)
    elif args.model == 'dcnv2':
        model = DCNv2(
            d_in=dataset.n_num_features, 
            categories=cats,
            d_out=d_out,
            **cfg_model
        ).to(device)
    elif args.model == 'saint':
        model = SAINT(
            d_numerical=dataset.n_num_features,
            categories=cats,
            d_out=d_out,
        ).to(device)
    """Optimizers"""
    if args.model in ['autoint', 'ftt']:
        def needs_wd(name):
            return all(x not in name for x in ['tokenizer', '.norm', '.bias'])

        parameters_with_wd = [v for k, v in model.named_parameters() if needs_wd(k)]
        parameters_without_wd = [v for k, v in model.named_parameters() if not needs_wd(k)]
        optimizer = make_optimizer(
            cfg_training['optimizer'],
            (
                [
                    {'params': parameters_with_wd},
                    {'params': parameters_without_wd, 'weight_decay': 0.0},
                ]
            ),
            cfg_training['lr'],
            cfg_training['weight_decay'],
        )
    else:
        optimizer = make_optimizer(
            cfg_training['optimizer'],
            model.parameters(),
            cfg_training['lr'],
            cfg_training['weight_decay'],
        )

    if args.model.startswith('tv7'):
        model.vec_batch_size = get_tv_vector_batch_size(model, optimizer, train_loader)
    best_val_score = train(model, optimizer)
    del model
    torch.cuda.empty_cache()
    gc.collect
    return best_val_score

def get_tv_vector_batch_size(model:TV7, optimizer, train_loader):
    loss_fn = (
        F.binary_cross_entropy_with_logits
        if dataset.is_binclass
        else F.cross_entropy
        if dataset.is_multiclass
        else F.mse_loss
    )
    # if args.model.startswith('tv') and args.model not in ['tv6.2', 'tv7.2']: # Not using for tv6.2
    #     loss_fn = lambda y,y_hat: F.hinge_embedding_loss(y, y_hat, margin=0.5)
    model.train()
    model.vec_batch_size = model.n_vectors
    while True:
        if model.vec_batch_size <= 0:
            print(f'ERROR: TV7 ran out of memory')
        try:
            for iteration, batch in enumerate(train_loader):
                if check_list['X_num']:
                    x_num = batch[0]
                    x_cat = None
                    if check_list['X_cat']:
                        x_cat = batch[1]
                else:
                    x_num = None
                    x_cat = batch[0]
                y = batch[-1]

                optimizer.zero_grad()
                logits = model(x_num, x_cat)
                if logits.ndim == 2:
                    logits = logits.squeeze(-1)
                loss = loss_fn(logits, y)
                if args.model.startswith('tv'):  # Earlier for tv6.4 this was not set to true
                    if model.reg_wt > 0:
                        loss += model.reg_wt * model.regularizer(use_low_gpu=False)
                loss.backward()
                optimizer.step()
                # tot_step += 1
                # tot_tr_loss += loss.cpu().item()
            break
        except RuntimeError:
            model.vec_batch_size = model.vec_batch_size//2
    return model.vec_batch_size

def train(model, optimizer):
    """Loss Function"""
    loss_fn = (
        F.binary_cross_entropy_with_logits
        if dataset.is_binclass
        else F.cross_entropy
        if dataset.is_multiclass
        else F.mse_loss
    )

    # using binary cross entropy was better than hinge loss
    # if args.model.startswith('tv') and args.model not in ['tv6.2', 'tv7.2']: # Not using for tv6.2
    #     loss_fn = lambda y,y_hat: F.hinge_embedding_loss(y, y_hat, margin=0.5)

    """Utils Function"""
    def apply_model(x_num, x_cat):
        logits = model(x_num, x_cat)
        if logits.ndim == 2:
            return logits.squeeze(-1)
        return logits

    @torch.inference_mode()
    def evaluate(parts):
        model.eval()
        results = {}
        for part in parts:
            assert part in ['train', 'val', 'test']
            golds, preds = [], []
            for batch in dataloaders[part]:
                if check_list['X_num']:
                    x_num = batch[0]
                    x_cat = None
                    if check_list['X_cat']:
                        x_cat = batch[1]
                else:
                    x_num = None
                    x_cat = batch[0]
                y = batch[-1]

                preds.append(apply_model(x_num, x_cat).cpu())
                golds.append(y.cpu())
            score = calculate_metrics(
                torch.cat(golds).numpy(),
                torch.cat(preds).numpy(),
                dataset.task_type.value,
                'logits' if not dataset.is_regression else None,
                dataset.y_info
            )[metric_key] * scale
            results[part] = score
        return results

    """Training"""
    best_metric = -np.inf
    no_improvement = 0

    for epoch in range(500):
        model.train()
        # for batch in tqdm(train_loader, desc=f'epoch-{epoch}'):
        for batch in train_loader:
            if check_list['X_num']:
                x_num = batch[0]
                x_cat = None
                if check_list['X_cat']:
                    x_cat = batch[1]
            else:
                x_num = None
                x_cat = batch[0]
            y = batch[-1]

            optimizer.zero_grad()
            loss = loss_fn(apply_model(x_num, x_cat), y)
            if args.model.startswith('tv'):
                if model.reg_wt > 0:
                    loss += model.reg_wt * model.regularizer(use_low_gpu=False)
            loss.backward()
            optimizer.step()

        scores = evaluate(['train', 'val', 'test'])
        val_score = scores['val']
        train_score = scores['train']
        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d} | Scores: Train= {train_score:.4f} | Validation= {val_score:.4f} | Train size= {len(train_loader.dataset)}')
        if val_score > best_metric:
            best_metric = val_score
            # print(' <<< BEST VALIDATION EPOCH')
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement == args.early_stop:
            print('early stop!')
            break
    return best_metric


cfg_model = model_param_spaces[args.model]
const_params = {
    p: v for p, v in cfg_model.items()
    if not isinstance(v, tuple)
}
cfg_training = training_param_spaces[args.model]
const_training_params = {
    p: v for p, v in cfg_training.items()
    if not isinstance(v, tuple)
}
cfg_file = f'{args.output}/cfg-tmp.json'
def save_per_iter(study, trial):
    saved_model_cfg = {**const_params}
    saved_training_cfg = {**const_training_params}
    for k in cfg_model:
        if k not in saved_model_cfg:
            saved_model_cfg[k] = study.best_trial.params.get(k)
            if k == 'use_kernel' and saved_model_cfg[k] == 'rbf':
                saved_model_cfg["gamma"] = study.best_trial.params.get("gamma")
    for k in cfg_training:
        if k not in saved_training_cfg:
            saved_training_cfg[k] = study.best_trial.params.get(k)
    saved_training_cfg = {
        'batch_size': batch_size,
        'eval_batch_size': val_batch_size,
        **saved_training_cfg
    }
    hyperparams = {
        'metric': metric_key,
        'eval_score': study.best_trial.value,
        'n_trial': study.best_trial.number,
        'dataset': args.dataset,
        'model': saved_model_cfg,
        'training': saved_training_cfg,
    }
    with open(cfg_file, 'w') as f:
        json.dump(hyperparams, f, indent=4, ensure_ascii=False)

iterations = 100
study = optuna.create_study(direction="maximize")
study.optimize(func=objective, n_trials=iterations, callbacks=[save_per_iter], gc_after_trial=True)


cfg_file = f'{args.output}/cfg.json'
for k in cfg_model:
    if k not in const_params:
        const_params[k] = study.best_params.get(k)
        if k == 'use_kernel' and const_params[k] == 'rbf':
            const_params["gamma"] = study.best_params.get("gamma")
for k in cfg_training:
    if k not in const_training_params:
        const_training_params[k] = study.best_params.get(k)
const_training_params = {
    'batch_size': batch_size,
    'eval_batch_size': val_batch_size,
    **const_training_params
}

hyperparams = {
    'metric': metric_key,
    'eval_score': study.best_value,
    'n_trial': study.best_trial.number,
    'dataset': args.dataset,
    'model': const_params,
    'training': const_training_params,
}
with open(cfg_file, 'w') as f:
    json.dump(hyperparams, f, indent=4, ensure_ascii=False)
