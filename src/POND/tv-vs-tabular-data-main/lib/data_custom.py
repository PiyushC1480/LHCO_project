import json
import torch
from pathlib import Path
import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms
from transformers import CLIPVisionModel, CLIPProcessor
from torch.utils.data import DataLoader, Dataset 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from typing import Dict, List, Literal, Optional, Union, Tuple
import pdb

from .data import Dataset2, TaskType

MNIST_VERSION = Literal['flat', 'clip']
CUSTOM_DATASET = Literal['mnist', 'cifar10']

class NumpyDataset(Dataset):
    def __init__(self, X, y,  transform=None, target_transform=None):
        self.img_labels = y
        self.imgs = X
        self.classes = set(y.flatten())
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.imgs[idx]#.reshape(-1,1)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def get_custom_db(checkpoint_dir, dataset_name: CUSTOM_DATASET, reduce_data:float = 0.8, preproc_type = 'ftt', seed = 42):

    with open(Path(checkpoint_dir) / 'data_config.json' ) as f:
        dconf = json.load(f)
    if dataset_name == 'mnist':
        return get_mnist(dconf, reduce_data, seed = seed, version = 'flat' )


def get_mnist(dconf, reduce_data = 0.8, seed = 42, version: MNIST_VERSION = 'flat'):
    '''
    dummy dconf:
    data:  ## For MNIST
        name: mnist
        num_classes: 2
        num_features: 784
        randomize: False
    '''
    multiply_test_size = 10
    if 'multiply_test_size' in dconf :
        multiply_test_size = dconf['multiply_test_size']
    val_size = 0.2
    if 'val_size' in dconf:
        val_size = dconf['val_size']
    rotate = False
    if 'rotate' in dconf:
        rotate = dconf['rotate']
    train_size = 0.1
    if 'train_size' in dconf:
        train_size = dconf['train_size']
    if version == 'flat':
        flatten = True
    else:
        flatten = False
        
    torch.manual_seed(seed) # to get same rotated images, but this will affect the models as well.
    if rotate:
        transform = transforms.Compose([transforms.RandomRotation(degrees=rotate),
                                   transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.ToTensor()])
    
    training_data = MNIST(
        root='data/mnist',
        train=True,
        download=True,
        transform=transform
    )
    testing_data = MNIST(
        root='data/mnist',
        train=False,
        download=True,
        transform=transform
    )

    npr = np.random.RandomState(seed)

    p_class = npr.randint(0,len(dconf['p_classes']))
    p_class = dconf['p_classes'][f"{p_class}"]
    
    X_train, y_train = filter_multilabel_data(training_data, p_class, npr=npr, flatten = flatten)
    X_test, y_test  = filter_multilabel_data(testing_data, p_class, npr=npr)
    
    if reduce_data > 0.0:
        X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=reduce_data, random_state=seed, stratify=y_train)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=seed, stratify=y_train)
    num_features = X_train.shape[1]

    ## increasing test_data size
    X_test , y_test = increase_size(X_test, y_test, times= multiply_test_size)

    print('Dataset details:')
    print(f'{X_train.shape = }, {X_val.shape = }, {X_test.shape = }')
    print(f'Positive points=> train= {y_train.sum()}, val= {y_val.sum()}, test= {y_test.sum()}')

    # if model_name in ['tv', 'tv2']:
    #     train_data =  NumpyDataset(X=X_train, y=y_train)
    #     val_data =  NumpyDataset(X=X_val, y=y_val)
    #     test_data = NumpyDataset(X=X_test, y= y_test)
    # elif model_name in ['fc', 'svm']:
    #     train_data = (X_train, y_train)
    #     val_data = (X_val, y_val)
    #     test_data = (X_test, y_test)

    # return train_data, val_data, test_data, num_features
    
    n_classes = 1
    task_type = 'binclass'
    X_num = {'train': X_train, 'test': X_test, 'val': X_val}
    X_cat = None
    X_str = None
    ys = {'train': y_train, 'test': y_test, 'val': y_val}
    return Dataset2(
        X_num,
        X_cat,
        X_str,
        ys,
        {},
        {}, # feature_names
        TaskType(task_type),
        n_classes,
        dconf['name'],
        None,
    )

def increase_size(X, y, times = 1):
    Large_X = X
    Large_y = y
    for i in range(times):
        Large_X = np.append(Large_X,X, axis=0)
        Large_y = np.append(Large_y,y, axis=0)
        
    return Large_X, Large_y

def get_openml(dconf, dataset_id, seed=42, reduce_data = 0.8, verbosity = 1)-> Dataset2:
    val_size = 0.2
    if 'val_size' in dconf:
        val_size = dconf['val_size']
    multiply_test_size = 10
    if 'multiply_test_size' in dconf:
        multiply_test_size = dconf['multiply_test_size']
    train_size = 0.1
    if 'train_size' in dconf:
        train_size = dconf['train_size']
    cat_nan_policy = 'most_frequent'
    if 'cat_nan_policy' in dconf:
        cat_nan_policy = dconf['cat_nan_policy']
    num_nan_policy = 'mean'
    if 'num_nan_policy' in dconf:
        num_nan_policy = dconf['num_nan_policy']
    cat_policy = 'indices'
    if 'cat_policy' in dconf:
        cat_policy = dconf['cat_policy']
    normalize = 'min_max'
    if 'normalize' in dconf:
        normalize = dconf['normalize']
    if seed is not None:
        npr = np.random.RandomState(seed)
    else:
        npr = np.random

    db = OpenMLDb.from_openml(dataset_id)
    if db.is_multiclass:
        raise Exception('Can\'t handle multicalss databases')
    dataset_len = len(db.X)
    stratify = None if db.is_regression else db.y
    train_idx, test_idx = train_test_split(range(dataset_len), random_state=seed, shuffle=dconf['shuffle'], stratify=stratify)
    
    # N,C = db.build_X(train_idx=train_idx,
    #                test_idx=test_idx,
    #                num_nan_policy=num_nan_policy,
    #                cat_nan_policy=cat_nan_policy,
    #                cat_policy=cat_policy,
    #                normalization='standard',
    #                seed=seed)
    if db.is_regression:
        y, info = db.build_y(train_idx=train_idx, test_idx=test_idx, policy='mean_std')
    else:
        y, info = db.build_y(train_idx=train_idx, test_idx=test_idx, policy=None)
    y= np.array(y, dtype=np.float32)
    

    if reduce_data > 0.0:
         stratify = None if db.is_regression else y[train_idx]
         train_idx, _ = train_test_split(train_idx, test_size=reduce_data, random_state=seed, stratify=stratify)

    N,C = db.build_X(train_idx=train_idx,
                   test_idx=test_idx,
                   num_nan_policy=num_nan_policy,
                   cat_nan_policy=cat_nan_policy,
                   cat_policy=cat_policy,
                   normalization='standard',
                   seed=seed)

    stratify = None if db.is_regression else y[train_idx]
    train_idx, val_idx = train_test_split(train_idx, test_size= val_size, random_state=seed, stratify=stratify)

    X_num = None
    X_cat = None
    X_str = None
    if N is not None:
        N = np.array(N, dtype=np.float32)
        X_num = {'train': N[train_idx], 'test': N[test_idx], 'val': N[val_idx]}
    if C is not None:
        # C = np.array(C, dtype=np.float32)
        X_cat = {'train': C[train_idx], 'test': C[test_idx], 'val': C[val_idx]}
    if dconf['cat_policy'] == 'ohe': # consider numerical features
        X_num = (
            X_cat 
            if X_num is None
            else {k: np.hstack(X_num[k], X_cat[k]) for k in X_num}
        )
        X_cat = None
    ys = {'train': y[train_idx], 'test': y[test_idx], 'val': y[val_idx]}

    ## increasing test_data size
    if multiply_test_size:
        if X_num is not None:
            X_num['test'] , ys_new = increase_size(X_num['test'], ys['test'], times= multiply_test_size)
        if X_cat is not None:
            X_cat['test'] , ys_new = increase_size(X_cat['test'], ys['test'], times= multiply_test_size)
        ys['test'] = ys_new

    if normalize == 'min_max':
        normalize_fn = MinMaxScaler()
    elif normalize == 'std':
        normalize_fn = StandardScaler()
    if normalize != False:
        if X_num is not None:
            X_num['train'] = normalize_fn.fit_transform(X_num['train'])
            X_num['val'] = normalize_fn.transform(X_num['val'])
            X_num['test'] = normalize_fn.transform(X_num['test'])
            # Not transforming categorical features
            # if X_cat is not None:
            #     X_cat['train'] = normalize_fn.fit_transform(X_cat['train'])
            #     X_cat['val'] = normalize_fn.transform(X_cat['val'])
            #     X_cat['test'] = normalize_fn.transform(X_cat['test'])

    if db.is_regression:
        task_type = "regression"
        n_classes = len(np.unique(ys['train']))
    else:
        task_type = "binclass"
        n_classes = 1

    # X_train, X_test = X[train_idx], X[test_idx]
    # y_train, y_test = y[train_idx], y[test_idx]

    # X_train, _ , y_train, _ = train_test_split(X_train, y_train, train_size=train_size, random_state=seed, stratify=y_train)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size= val_size, random_state=seed, stratify=y_train)
    # num_features = X_train.shape[1]
    

    if verbosity >= 1:
        print('Dataset details:')
        get_shapes = lambda part: (
            X_num[part].shape[0] if X_num is not None else X_cat[part].shape[0],
            X_num[part].shape[1] if X_num is not None else 0 + X_cat[part].shape[1] if X_cat is not None else 0
        )
        print(f'\tShapes => train= {get_shapes("train")}, val= {get_shapes("val")}, test= {get_shapes("test")}')
        print(f'\tPositive points=> train= {ys["train"].sum()}, val= {ys["val"].sum()}, test= {ys["test"].sum()}')

    # if return_info:
    #     return train_data, val_data, test_data, num_features, db.info
    return Dataset2(
        X_num,
        X_cat,
        X_str,
        ys,
        {},
        {}, # feature_names
        TaskType(task_type),
        n_classes,
        dconf['name'],
        None,
    )



def filter_multilabel_data(data, p_class = 'airplane', p_cls_size = -1, npr = None, flatten = True):
    '''
    Filtering like done in FROCC paper 
    '''
    X = []
    y = []
    dataset_classes = data.classes
    points_idx = {key: [] for key in dataset_classes}
    for idx, (_, label) in enumerate(data):
        points_idx[dataset_classes[label]].append(idx)

    if p_cls_size == -1 or p_cls_size > len(points_idx[p_class]):
        p_cls_size = len(points_idx[p_class])
    
    positive_points = []
    negative_points = []
    for cls in dataset_classes:
        if cls != p_class:
            points_idx[cls] = npr.choice(np.array(points_idx[cls]), p_cls_size//(len(dataset_classes)-1), replace=False).tolist()
            negative_points += points_idx[cls]
        else:
            points_idx[cls] = npr.choice(np.array(points_idx[cls]), p_cls_size, replace=False).tolist()
            positive_points += points_idx[cls]

    for idx, (img, label) in enumerate(data):
        if idx in positive_points:
            y.append(1)
            if flatten:
                X.append(img.numpy().flatten())
            else:
                X.append(img.numpy())
        elif idx in negative_points:
            y.append(0)
            if flatten:
                X.append(img.numpy().flatten())
            else:
                X.append(img.numpy())
        # else: skipping that points

    return np.array(X), np.array(y, dtype=np.float32)
