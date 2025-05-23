import pdb
from numpy.linalg import test
import yaml
from typing import Dict, Any
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
import json
from types import SimpleNamespace

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_blobs, make_moons
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms
# from torchvision.datasets import MNIST, CIFAR10, CIFAR100, Omniglot
# from transformers import CLIPVisionModel, CLIPProcessor

import openml
# from openml.datasets import get_dataset
from .data_utils_openml import Dataset as OpenMLDb
from .data import Dataset2, TaskType
# from torchvision.transforms import ToTensor

def get_config(config_path = 'config.yaml')-> Dict:
    with open(config_path, 'r') as f:
        contents = yaml.safe_load(f)
    contents = json.loads(json.dumps(contents), object_hook = lambda x: SimpleNamespace(**x))
    return contents

def config_to_dict(config):
    return {
        k: config_to_dict(v) if isinstance(v, SimpleNamespace) else v for k,v in vars(config).items()
    }

def save_config(config, path = './results/config.yaml'):
    config_dict = config_to_dict(config)
    with open(path, '+w') as f:
        yaml.dump(config_dict, f)

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

def old_get_openml_db(model_name = 'tv', dataset_name = 'cifar10', seed = 1234, return_info = False):
    '''
    model_name: str ; Name of the model
    dataset_name: str or result of get_config for data
    ---
    returns dataset_dict: numpy or pytorch dataset
    '''
    ## Prefeer not to config for dataset and model here, as they could have changed 
    if type(dataset_name) == str:
        config_path = './configs/data/' + dataset_name + '.yaml'
        dconf = get_config(config_path)
    else:
        dconf = dataset_name
        dataset_name = dconf.name
    
    if dataset_name != dconf.name:
        raise Exception(f'Dataset name mismatch: {dataset_name} != {dconf.name}')
    
    if model_name.startswith('tv') or model_name in ['mlp']:
        model_name = 'tv'
    if model_name in ['xgb', 'svm', 'fc']:
        model_name = 'fc'

    if dataset_name == 'mnist':
        dataset_dict = get_mnist(dconf, model_name, seed = seed)
    elif dataset_name == 'cifar10':
        dataset_dict = get_cifar10(dconf, model_name, seed = seed)
    elif dataset_name == 'cifar100':
        dataset_dict = get_cifar100(dconf, model_name, seed = seed)
    elif dataset_name == 'magic':
        dataset_dict = get_magic_gt_dataset(dconf, model_name, seed = seed)
    elif dataset_name == 'miniBoo':
        dataset_dict = get_miniBoo(dconf, model_name, seed = seed)
    elif dataset_name == 'omniglot':
        dataset_dict = get_omniglot(dconf, model_name, seed = seed)
    elif dataset_name == 'blob_outlier':
        dataset_dict = get_blob_outlier(dconf, model_name, seed)
    elif dataset_name == 'embd_cifar':
        dataset_dict = get_embd_cifar10(dconf, model_name, seed)
    elif dataset_name == 'moon':
        dataset_dict = get_moon(dconf, model_name, seed = seed)
    elif dataset_name == 'blob':
        dataset_dict = get_blobs(dconf, model_name, seed = seed)
    elif dataset_name == 'higsBoson':
        dataset_dict = get_higs_boson(dconf, model_name, seed = seed)
    elif dataset_name.startswith('openml'):
        # dataset_dict = get_openml_single_db(dconf, model_name, seed = seed)
        dataset_dict = get_openmlcc18(dconf, model_name, seed = seed, return_info = return_info)
    else:
        raise Exception('#'*20+' No dataset found for specified config '+'#'*20)
    
    return dataset_dict

def get_openml_db(checkpoint_dir, fientune_dir, dataset_id, reduce_data = 0.8, preproc_type='ftt', seed = 42, verbosity = 1):
    '''
    model_name: str ; Name of the model
    dataset_name: str or result of get_config for data
    ---
    returns dataset_dict: numpy or pytorch dataset
    '''
    # setting feature_names = {} works for openml binary data
    with open(Path(checkpoint_dir) / 'data_config.json' ) as f:
        dconf = json.load(f)

    if preproc_type == 'ftt':
        dconf['cat_policy'] = 'indices'
    elif preproc_type == 'xgboost':
        dconf['cat_policy'] = 'ohe'
    elif preproc_type == 'catboost':
        dconf['cat_policy'] = 'indices'

    dataset_dict = get_openmlcc18(dconf = dconf,dataset_id = dataset_id, seed = seed, reduce_data = reduce_data, verbosity = verbosity)
    return dataset_dict


#################### Bacis datasets, but depricated code for model #################### 
# def get_blobs(datapoints = 200, n_features=2, centers=2, random_state=123, split_size=0.33):
#     X_data, y_data = make_blobs(datapoints, n_features=n_features, centers=centers, random_state=random_state)
#     X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=split_size, random_state=random_state)
#     min_max_scaler = MinMaxScaler()
#     X_train = min_max_scaler.fit_transform(X_train)
#     X_test = min_max_scaler.transform(X_test)
#     
#     return X_train, X_test, y_train, y_test

def get_blobsh(datapoints = 200, n_features=2, centers=2, random_state=123, split_size=0.33, dimension=2):
    
    small_deviation = 0.1
    X_data, y_data = make_blobs(datapoints, n_features=n_features, centers=centers, random_state=random_state)

    if centers>2:
        r_np = np.random.RandomState(random_state)
        for c in range(centers):
            y_data[y_data == c] = 0 if r_np.rand()>0.5 else 1

    
    if dimension>n_features:
        dummy_data = np.zeros((X_data.shape[0], dimension - X_data.shape[1])) + 2
        X_data = np.append(X_data, dummy_data, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=split_size, random_state=random_state)
    min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)
    
    X_train[:,n_features:] = small_deviation
    X_test[:, n_features:] = small_deviation
    
    dataset_dict = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}
    return dataset_dict

def get_classification(datapoints=200, n_features=2, centers=2, random_state=123, split_size=0.33):
    
    if n_features <=3:
        X_data, y_data = make_classification(datapoints, n_features=n_features, n_redundant=0, n_repeated=0, random_state=random_state)
    else:
        X_data, y_data = make_classification(datapoints, n_features=n_features, random_state=random_state)
    
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=split_size, random_state=random_state)
    min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test 

def get_moon(data_config, model_name, seed):
        
    datapoints=data_config.datapoints
    noise = data_config.noise
    random_state= seed
    split_size= data_config.split_size
    
    X_data, y_data = make_moons(datapoints, noise=noise, random_state=random_state)
    X_data = np.array(X_data, dtype=np.float32)
    y_data = np.array(y_data, dtype=np.float32)
    
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=split_size, random_state=random_state)
    min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)

    if model_name in ['tv', 'tv2']:
        train_data =  NumpyDataset(X=X_train, y=y_train)
        test_data = NumpyDataset(X=X_test, y= y_test)
    elif model_name in ['fc', 'svm']:
        train_data = (X_train, y_train)
        test_data = (X_test, y_test)

    return train_data, test_data

def get_blobs(data_config, model_name, seed):
        
    datapoints=data_config.datapoints
    noise = data_config.noise
    random_state = seed
    split_size= data_config.split_size
    cordinate_1 = data_config.cordinate_1
    cordinate_2 = data_config.cordinate_2
    
    # X_data, y_data = make_blobs(datapoints, n_features=2, centers=2, random_state=random_state)
    npr = np.random.RandomState(random_state)
    X_data_1 = [npr.uniform(cordinate_1[0],cordinate_1[2], datapoints//2),npr.uniform(cordinate_1[1],cordinate_1[3], datapoints//2)]
    X_data_2 = [npr.uniform(cordinate_2[0],cordinate_2[2], datapoints//2),npr.uniform(cordinate_2[1],cordinate_2[3], datapoints//2)]
    X_data = np.concatenate([np.array(X_data_1).T, np.array(X_data_2).T], axis=0)
    y_data = np.concatenate([np.zeros(datapoints//2), np.ones(datapoints//2)], axis=0)
    print(f'{X_data.shape=}, {y_data.shape=}')
    
    X_data = np.array(X_data, dtype=np.float32)
    y_data = np.array(y_data, dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=split_size, random_state=random_state)

    normalize = 'min_max' # other option 'std'
    if hasattr(data_config, 'normalize'):
        normalize = data_config.normalize
    
    if normalize == 'min_max':
        normalize_fn = MinMaxScaler()
    elif normalize == 'std':
        normalize_fn = StandardScaler()
    if normalize != False:
        X_train = normalize_fn.fit_transform(X_train)
        X_test = normalize_fn.transform(X_test)
    

    if model_name in ['tv', 'tv2']:
        train_data =  NumpyDataset(X=X_train, y=y_train)
        test_data = NumpyDataset(X=X_test, y= y_test)
    elif model_name in ['fc', 'svm']:
        train_data = (X_train, y_train)
        test_data = (X_test, y_test)

    return train_data, test_data

def get_moonh(datapoints=200, dimension=200, noise = 0.1, random_state=123, split_size=0.33, transform = False):
    
    small_deviation = 0.1
    if dimension<2:
        raise Exception('dimension should be >=2')
        
    X_data, y_data = make_moons(datapoints, noise=noise, random_state=random_state)
    X_data = np.append(np.zeros((X_data.shape[0], dimension - X_data.shape[1]))+5, X_data, 1)

    if transform:
        r_np = np.random.RandomState(random_state)
        Transform_matrix = r_np.rand(dimension, dimension) 
        X_data = np.dot(X_data, Transform_matrix)
    
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=split_size, random_state=random_state)
    min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)
    
    X_train[:,2:] += small_deviation
    X_test[:, 2:] += small_deviation
    
    dataset_dict = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}
    return dataset_dict
####################################### Basic datasets ends #############################3333
    
def get_mnist(data_config, model_name, seed):
    '''
    dummy data_config:
    data:  ## For MNIST
        name: mnist
        num_classes: 2
        num_features: 784
        randomize: False
    '''
    multiply_test_size = 10
    if hasattr(data_config, 'multiply_test_size'):
        multiply_test_size = data_config.multiply_test_size
    val_size = 0.2
    if hasattr(data_config, 'val_size'):
        val_size = data_config.val_size
    rotate = True
    if hasattr(data_config, 'rotate'):
        rotate = data_config.rotate
        
    torch.manual_seed(seed) # to get same rotated images, but this will affect the models as well.
    if rotate:
        transform = transforms.Compose([transforms.RandomRotation(degrees=50),
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

    ## To support old code, for randomize
    if hasattr(data_config, 'randomize'):
        randomize = data_config.randomize
    else:
        randomize = False

    npr = np.random.RandomState(seed)

    p_class = npr.randint(0,len(data_config.p_classes))
    p_class = data_config.p_classes[p_class]
    
    X_train, y_train = filter_multilabel_data(training_data, p_class, npr=npr)
    X_test, y_test  = filter_multilabel_data(testing_data, p_class, npr=npr)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=seed, stratify=y_train)
    num_features = X_train.shape[1]
    
    ## increasing test_data size
    X_test , y_test = increase_size(X_test, y_test, times= multiply_test_size)

    print('Dataset details:')
    print(f'{X_train.shape = }, {X_val.shape = }, {X_test.shape = }')
    print(f'Positive points=> train= {y_train.sum()}, val= {y_val.sum()}, test= {y_test.sum()}')

    if model_name in ['tv', 'tv2']:
        train_data =  NumpyDataset(X=X_train, y=y_train)
        val_data =  NumpyDataset(X=X_val, y=y_val)
        test_data = NumpyDataset(X=X_test, y= y_test)
    elif model_name in ['fc', 'svm']:
        train_data = (X_train, y_train)
        val_data = (X_val, y_val)
        test_data = (X_test, y_test)

    return train_data, val_data, test_data, num_features


# def filter_mnist_data(data, num_classes = 2, randomize = False):
#     '''
#     Depricated
#     '''
#     X = []
#     y = []
#     for img, label in data:
#         if label < num_classes:
#             img = img.numpy().flatten()
#             X.append(img)
#             if randomize:
#                 r_label = np.random.randint(0,2)
#                 y.append(r_label)
#             else:
#                 y.append(label)
#     return np.array(X), np.array(y, dtype=np.float32)


def get_cifar10(data_config, model_name, seed):
    '''
    dummy data_config:
    data: 
        name: cifar10
        p_classes: [airplane, dear] 
        num_features: 3072
    '''
    multiply_test_size = 10
    if hasattr(data_config, 'multiply_test_size'):
        multiply_test_size = data_config.multiply_test_size
    val_size = 0.2
    if hasattr(data_config, 'val_size'):
        val_size = data_config.val_size
    rotate = True
    if hasattr(data_config, 'rotate'):
        rotate = data_config.rotate
    
    torch.manual_seed(seed) # to get same rotated images, but this will affect the models as well.
    if rotate:
        transform = transforms.Compose([transforms.RandomRotation(degrees=50),
                                   transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.ToTensor()])

    training_data = CIFAR10(
        root='data/cifar10',
        train=True,
        download=True,
        transform=transform
    )
    testing_data = CIFAR10(
        root='data/cifar10',
        train=False,
        download=True,
        transform=transform
    )

    npr = np.random.RandomState(seed)
    p_class = npr.randint(0,len(data_config.p_classes))
    p_class = data_config.p_classes[p_class]
    print('Selecting positive class as:', p_class)

    X_train, y_train = filter_multilabel_data(training_data, p_class, npr = npr)
    X_test, y_test  = filter_multilabel_data(testing_data, p_class, npr=npr)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=seed, stratify=y_train)
    num_features = X_train.shape[1]
    
    ## increasing test_data size
    X_test , y_test = increase_size(X_test, y_test, times= multiply_test_size)

    normalize = 'std' 
    if hasattr(data_config, 'normalize'):
        normalize = data_config.normalize
    
    if normalize == 'min_max':
        normalize_fn = MinMaxScaler()
    elif normalize == 'std':
        normalize_fn = StandardScaler()
    if normalize != False:
        X_train = normalize_fn.fit_transform(X_train)
        X_val = normalize_fn.transform(X_val)
        X_test = normalize_fn.transform(X_test)

    print('Dataset details:')
    print(f'{X_train.shape=}, {X_val.shape = }, {X_test.shape=}')
    print(f'Positive points=> train= {y_train.sum()}, val = {y_val.sum()}, test= {y_test.sum()}')

    if model_name in ['tv', 'tv2']:
        train_data =  NumpyDataset(X=X_train, y=y_train)
        val_data = NumpyDataset(X=X_val, y= y_val)
        test_data = NumpyDataset(X=X_test, y= y_test)
    elif model_name in ['fc', 'svm']: # for sklearn style simple numpy dataset
        train_data = (X_train, y_train)
        val_data = (X_val, y_val)
        test_data = (X_test, y_test)
    elif model_name in ['fc_occ']:
        train_data = (X_train[y_train == 1], y_train[y_train == 1])
        val_data  = (X_val, y_val)
        test_data  = (X_test, y_test)
    
    return train_data, val_data, test_data, num_features

def get_encoded_data(data, processor, encoder):
    len = data.shape[0]
    embd = []
    for img in tqdm(data, desc='Generating embdding'):
        inputs = processor(text=None, images= img, return_tensors='pt', do_rescale = False)
        with torch.no_grad():
            outputs= encoder(**inputs)
        embd.append(outputs.pooler_output.flatten()) # appending element shape: [1,768]

    
    return np.array(embd) #.view(len, -1)



def get_embd_cifar10(data_config, model_name, seed):
    '''
    dummy data_config:
    data: 
        name: cifar10
        p_classes: [airplane, dear] 
        num_features: 3072
    '''
    multiply_test_size = 10
    if hasattr(data_config, 'multiply_test_size'):
        multiply_test_size = data_config.multiply_test_size
    val_size = 0.2
    if hasattr(data_config, 'val_size'):
        val_size = data_config.val_size
    rotate = True
    if hasattr(data_config, 'rotate'):
        rotate = data_config.rotate
    train_size = 1.0
    if hasattr(data_config, 'train_size'):
        train_size = data_config.train_size
    
    torch.manual_seed(seed) # to get same rotated images, but this will affect the models as well.
    if rotate:
        transform = transforms.Compose([transforms.RandomRotation(degrees=50),
                                   transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.ToTensor()])

    training_data = CIFAR10(
        root='data/cifar10',
        train=True,
        download=True,
        transform=transform
    )
    testing_data = CIFAR10(
        root='data/cifar10',
        train=False,
        download=True,
        transform=transform
    )

    npr = np.random.RandomState(seed)
    p_class = npr.randint(0,len(data_config.p_classes))
    p_class = data_config.p_classes[p_class]
    print('Selecting positive class as:', p_class)

    X_train, y_train = filter_multilabel_data(training_data, p_class, npr = npr, flatten=False)
    X_test, y_test  = filter_multilabel_data(testing_data, p_class, npr=npr, flatten=False)
    
    # Selecting subset of dataseet for training
    # print(f'{train_size = }')
    X_train, _ , y_train, _ = train_test_split(X_train, y_train, train_size=train_size, random_state=seed, stratify=y_train)

    select_model = 'openai/clip-vit-base-patch32'
    processor = CLIPProcessor.from_pretrained(select_model)
    encoder = CLIPVisionModel.from_pretrained(select_model)
    X_train = get_encoded_data(X_train, processor, encoder)
    X_test = get_encoded_data(X_test, processor, encoder)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=seed, stratify=y_train)
    num_features = X_train.shape[1]
    
    ## increasing test_data size
    X_test , y_test = increase_size(X_test, y_test, times= multiply_test_size)

    normalize = 'std' 
    if hasattr(data_config, 'normalize'):
        normalize = data_config.normalize
    
    if normalize == 'min_max':
        normalize_fn = MinMaxScaler()
    elif normalize == 'std':
        normalize_fn = StandardScaler()
    if normalize != False:
        X_train = normalize_fn.fit_transform(X_train)
        X_val = normalize_fn.transform(X_val)
        X_test = normalize_fn.transform(X_test)

    print('Dataset details:')
    print(f'{X_train.shape=}, {X_val.shape = }, {X_test.shape=}')
    print(f'Positive points=> train= {y_train.sum()}, val = {y_val.sum()}, test= {y_test.sum()}')

    if model_name in ['tv', 'tv2']:
        train_data =  NumpyDataset(X=X_train, y=y_train)
        val_data = NumpyDataset(X=X_val, y= y_val)
        test_data = NumpyDataset(X=X_test, y= y_test)
    elif model_name in ['fc', 'svm']: # for sklearn style simple numpy dataset
        train_data = (X_train, y_train)
        val_data = (X_val, y_val)
        test_data = (X_test, y_test)
    elif model_name in ['fc_occ']:
        train_data = (X_train[y_train == 1], y_train[y_train == 1])
        val_data  = (X_val, y_val)
        test_data  = (X_test, y_test)
    
    return train_data, val_data, test_data, num_features


def get_cifar100(data_config, model_name, seed):
    '''
    dummy data_config:
    data: 
        name: cifar100
        p_classes: [beaver, motorcycle, fox] 
        num_features: 3072
    '''
    
    multiply_test_size = 10
    if hasattr(data_config, 'multiply_test_size'):
        multiply_test_size = data_config.multiply_test_size
    val_size = 0.2
    if hasattr(data_config, 'val_size'):
        val_size = data_config.val_size
    rotate = True
    if hasattr(data_config, 'rotate'):
        rotate = data_config.rotate
        
    torch.manual_seed(seed) # to get same rotated images, but this will affect the models as well.
    if rotate:
        transform = transforms.Compose([transforms.RandomRotation(degrees=50),
                                   transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.ToTensor()])

    training_data = CIFAR100(
        root='data/cifar100',
        train=True,
        download=True,
        transform=transform
    )
    testing_data = CIFAR100(
        root='data/cifar100',
        train=False,
        download=True,
        transform=transform
    )

    npr = np.random.RandomState(seed)
    p_class = npr.randint(0,len(data_config.p_classes))
    p_class = data_config.p_classes[p_class]
    print('Selecting positive class as:', p_class)

    X_train, y_train = filter_multilabel_data(training_data, p_class, data_config.p_cls_size, npr=npr)
    X_test, y_test  = filter_multilabel_data(testing_data, p_class, data_config.p_cls_size, npr=npr)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=seed, stratify=y_train)
    num_features = X_train.shape[1]
    
    ## increasing test_data size
    X_test , y_test = increase_size(X_test, y_test, times= multiply_test_size)
    
    normalize = 'std' 
    if hasattr(data_config, 'normalize'):
        normalize = data_config.normalize
    
    if normalize == 'min_max':
        normalize_fn = MinMaxScaler()
    elif normalize == 'std':
        normalize_fn = StandardScaler()
    if normalize != False:
        X_train = normalize_fn.fit_transform(X_train)
        X_val = normalize_fn.transform(X_val)
        X_test = normalize_fn.transform(X_test)

    print('Dataset details:')
    print(f'{X_train.shape=}, {X_val.shape = }, {X_test.shape=}')
    print(f'Positive points=> train= {y_train.sum()}, val = {y_val.sum()}, test= {y_test.sum()}')

    if model_name in ['tv', 'tv2']:
        train_data =  NumpyDataset(X=X_train, y=y_train)
        val_data = NumpyDataset(X=X_val, y= y_val)
        test_data = NumpyDataset(X=X_test, y= y_test)
    elif model_name in ['fc', 'svm']:
        train_data = (X_train, y_train)
        val_data = (X_val, y_val)
        test_data = (X_test, y_test)
    
    return train_data, val_data, test_data, num_features

def filter_cifar_data_old(data, classes= ['dog','cat']):
    X = []
    y = []
    my_class_to_idx = {key: value for value, key in enumerate(classes)}
    dataset_classes = data.classes
    # print('my class to idx')
    # print(my_class_to_idx)

    class_ids = [data.class_to_idx[cls] for cls in classes]
    for img, label in data:
        if label in class_ids:
            img = img.numpy().flatten()
            X.append(img)
            y.append(my_class_to_idx[dataset_classes[label]])
    return np.array(X), np.array(y, dtype=np.float32)

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

def get_magic_gt_dataset(data_config, model_name, seed):
    '''
    To return MAGIC gamma Telescope dataset
    Parameters:
        data_config: 
                        test_size: 
                        shuffle:

    '''
    val_size = 0.2
    if hasattr(data_config, 'val_size'):
        val_size = data_config.val_size
    df = pd.read_csv('./data/magic_gamma_telescope/magic04.data', header=None)
    df.iloc[:,-1] = 1.0*(df.iloc[:,-1] == 'g')
    y = df.iloc[:,-1].to_numpy(dtype=np.float32)
    X = df.iloc[:,:-1].to_numpy(dtype=np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = data_config.test_size, shuffle = data_config.shuffle, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = val_size, shuffle = data_config.shuffle, random_state=seed)
    num_features = X_train.shape[1]

    normalize = 'min_max' # other option 'std'
    if hasattr(data_config, 'normalize'):
        normalize = data_config.normalize
    
    if normalize == 'min_max':
        normalize_fn = MinMaxScaler()
    elif normalize == 'std':
        normalize_fn = StandardScaler()

    if normalize != False:
        X_train = normalize_fn.fit_transform(X_train)
        X_val = normalize_fn.transform(X_val)
        X_test = normalize_fn.transform(X_test)

    print('Dataset details:')
    print(f'{X_train.shape=}, {X_val.shape = }, {X_test.shape=}')
    print(f'Positive points=> train= {y_train.sum()}, val= {y_val.sum()}, test= {y_test.sum()}')

    if model_name in ['tv', 'tv2']:
        train_data =  NumpyDataset(X=X_train, y=y_train)
        val_data = NumpyDataset(X=X_val, y= y_val)
        test_data = NumpyDataset(X=X_test, y= y_test)
    elif model_name in ['fc', 'svm']:
        train_data = (X_train, y_train)
        val_data = (X_val, y_val)
        test_data = (X_test, y_test)

    return train_data, val_data, test_data, num_features

def get_miniBoo(data_config, model_name, seed):
    val_size = 0.2
    if hasattr(data_config, 'val_size'):
        val_size = data_config.val_size

    data_file = 'data/miniBooNE/MiniBooNE_PID.txt'
    with open(data_file) as f:
        line = f.readline()
    print('line', line)
    p_class_size, n_class_size = [int(i) for i in line.strip().split(' ')]
    # p_class_size, n_class_size

    dataset_df = pd.read_csv(data_file, skiprows=1, header=None, delim_whitespace=True)
    X = dataset_df.to_numpy(dtype=np.float32)
    y = np.append(np.ones(p_class_size, dtype=np.float32), np.zeros(n_class_size,dtype=np.float32), axis=0)

    ## selecting random split for dataset
    dataset_size_ratio = 1.0 - data_config.dataset_size/len(y)
    X, _, y, _ = train_test_split(X,y, test_size = dataset_size_ratio, shuffle = data_config.shuffle, random_state=seed)

    ## Note that shuffle is always true here
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = data_config.test_size, random_state=seed)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = val_size, random_state=seed)
    num_features = X_train.shape[1]

    print('Dataset details:')
    print(f'{X_train.shape=}, {X_val.shape = }, {X_test.shape=}')
    print(f'Positive points=> train= {y_train.sum()}, val = {y_val.sum()}, test= {y_test.sum()}')
    
    normalize = 'min_max' # other option 'std'
    if hasattr(data_config, 'normalize'):
        normalize = data_config.normalize
    
    if normalize == 'min_max':
        normalize_fn = MinMaxScaler()
    elif normalize == 'std':
        normalize_fn = StandardScaler()
    if normalize != False:
        X_train = normalize_fn.fit_transform(X_train)
        X_val = normalize_fn.transform(X_val)
        X_test = normalize_fn.transform(X_test)

    if model_name in ['tv', 'tv2']:
        train_data =  NumpyDataset(X=X_train, y=y_train)
        val_data = NumpyDataset(X=X_val, y= y_val)
        test_data = NumpyDataset(X=X_test, y= y_test)
    elif model_name in ['fc', 'svm']:
        train_data = (X_train, y_train)
        val_data = (X_val, y_test)
        test_data = (X_test, y_test)
 
    return train_data, val_data, test_data, num_features

def get_omniglot(data_config, model_name, seed):

    val_size = 0.2
    if hasattr(data_config, 'val_size'):
        val_size = data_config.val_size

    full_dataset = Omniglot(
        root='data/omniglot',
        background=True,
        download=True,
        transform=transforms.ToTensor()
    )
    p_classes = range(20)
    n_classes = range(20, 40)
    X, y = filter_omniglot(full_dataset,p_classes, n_classes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= data_config.test_size, random_state=seed)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size= val_size, random_state=seed)
    num_features = X_train.shape[1]

    print('Dataset details:')
    print(f'{X_train.shape=}, {X_val.shape = }, {X_test.shape=}')
    print(f'Positive points=> train= {y_train.sum()}, val = {y_val.sum()}, test= {y_test.sum()}')

    normalize = 'min_max' # other option 'std'
    if hasattr(data_config, 'normalize'):
        normalize = data_config.normalize
    
    if normalize == 'min_max':
        normalize_fn = MinMaxScaler()
    elif normalize == 'std':
        normalize_fn = StandardScaler()
    if normalize != False:
        X_train = normalize_fn.fit_transform(X_train)
        X_val = normalize_fn.transform(X_val)
        X_test = normalize_fn.transform(X_test)

    if model_name in ['tv', 'tv2']:
        train_data =  NumpyDataset(X=X_train, y=y_train)
        val_data = NumpyDataset(X=X_val, y= y_val)
        test_data = NumpyDataset(X=X_test, y= y_test)
    elif model_name in ['fc', 'svm']:
        train_data = (X_train, y_train)
        val_data = (X_val, y_val)
        test_data = (X_test, y_test)
 
    return train_data, val_data, test_data, num_features

def filter_omniglot(full_dataset, p_classes, n_classes):
    positive_points = []
    negative_points = []
    for idx, (img, label) in enumerate(full_dataset):
        if label in p_classes:
            positive_points.append(idx)
        elif label in n_classes:
            negative_points.append(idx)
    
    # filtering only picked labels
    X, y = [], []
    for idx, (img, label) in enumerate(full_dataset):
        if idx in positive_points:
            X.append(img.numpy().flatten())
            y.append(1)
        elif idx in negative_points:
            X.append(img.numpy().flatten())
            y.append(0)

    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    return X,y

def get_blob_outlier(dconf, model_name, seed):

    X, y, X_o, y_o = make_outlier_db(n_samples=dconf.n_samples, 
                           outlier_ratio=dconf.outlier_ratio, 
                           class_sep=dconf.class_sep, 
                           std = dconf.std, seed=seed)

    print(f'{X.shape = }, {y.shape = }')
    val_size = 0.2
    if hasattr(dconf, 'val_size'):
        val_size = dconf.val_size

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= dconf.test_size, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size= val_size, random_state=seed)
    
    X_train = np.concatenate((X_train, X_o), axis=0, dtype=np.float32)
    y_train = np.concatenate((y_train, y_o), axis=0)
    num_features = X_train.shape[1]

    print('Dataset details:')
    print(f'{X_train.shape=}, {X_val.shape = }, {X_test.shape=}')
    print(f'Positive points=> train= {y_train.sum()}, val = {y_val.sum()}, test= {y_test.sum()}')

    normalize = 'std' # other option 'min_max'
    if hasattr(dconf, 'normalize'):
        normalize = dconf.normalize
    
    if normalize == 'min_max':
        normalize_fn = MinMaxScaler()
    elif normalize == 'std':
        normalize_fn = StandardScaler()
    if normalize != False:
        X_train = normalize_fn.fit_transform(X_train)
        X_val = normalize_fn.transform(X_val)
        X_test = normalize_fn.transform(X_test)

    if model_name in ['tv', 'tv2']:
        train_data =  NumpyDataset(X=X_train, y=y_train)
        val_data = NumpyDataset(X=X_val, y= y_val)
        test_data = NumpyDataset(X=X_test, y= y_test)
    elif model_name in ['fc', 'svm']:
        train_data = (X_train, y_train)
        val_data = (X_val, y_val)
        test_data = (X_test, y_test)
 
    return train_data, val_data, test_data, num_features
    

def make_outlier_db(n_samples, outlier_ratio = 0.1, class_sep = 2, std = 1, seed = None, shuffle = True, add_outlier_to_train = True):
    n_outliers = int(n_samples * outlier_ratio)
    n_samples = int(n_samples - n_outliers)
    p_center = np.array([1,1])
    n_center = np.array((p_center[0] + class_sep * np.sqrt(2),p_center[1] + class_sep*np.sqrt(2)))
    if seed is not None:
        npr = np.random.RandomState(seed)
    else:
        npr = np.random
    X_p = npr.normal(loc=p_center, scale= (std, std), size=(n_samples//2, 2))
    X_n = npr.normal(loc=n_center, scale= (std, std), size=(n_samples//2, 2))
    
    ## Adding outliers
    o_center = np.array(p_center + n_center)/2 #+ p_center + n_center
    o_std = std/5
    # print(f'{o_center = }, {o_std = }')
    X_o = npr.normal(loc=o_center, scale=(o_std,o_std), size=(n_outliers, 2))
    y_o = np.zeros(n_outliers)
    for i in range(n_outliers):
        dist_from_n = np.linalg.norm(X_o[i] - n_center)
        dist_from_p = np.linalg.norm(X_o[i] - p_center)
        if dist_from_n < dist_from_p:
            y_o[i] = 1
    
    
    if add_outlier_to_train:
        X = np.concatenate((X_p, X_n), axis=0, dtype=np.float32)
        y = np.concatenate((np.ones(n_samples//2), np.zeros(n_samples//2)), axis=0)
    else:
        X = np.concatenate((X_p, X_n, X_o), axis=0, dtype=np.float32)
        y = np.concatenate((np.ones(n_samples//2), np.zeros(n_samples//2), y_o), axis=0)
    # print(X.shape, y.shape)
    
    if shuffle:
        idx = np.arange(X.shape[0])
        npr.shuffle(idx)
        X = X[idx]
        y = y[idx]
        
    return X,y, X_o, y_o

def get_clip(dconf, model_name, seed):
    pass

def get_higs_boson(dconf, model_name, seed):

    val_size = 0.2
    if hasattr(dconf, 'val_size'):
        val_size = dconf.val_size
    root_path = './data/HigsBoson/'
    if hasattr(dconf, 'root_path'):
        root_path = dconf.root_path
    train_size = 0.1
    if hasattr(dconf, 'train_size'):
        train_size = dconf.train_size

    training_data = pd.read_csv(root_path+'training.csv')
    test_data = pd.read_csv(root_path+'test.csv')

    ## Preprocess training data
    training_data.drop('EventId', axis=1, inplace=True)
    y = pd.DataFrame(training_data['Label'])
    le = LabelEncoder()
    y = le.fit_transform(y['label'])
    X = training_data.drop(['Label', 'Weight'], axis = 1)
    ## Proprocess testing data
    test_data.drop('EventId', axis=1, inplace=True)
    y_test = pd.DataFrame(test_data['Label'])
    y_test = le.transform(y_test['label'])
    X_test = test_data.drop(['Label', 'Weight'], axis = 1).values

    ## Selecting subset of training data
    X, _ , y, _ = train_test_split(X_train, y_train, train_size=train_size, random_state=seed, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size= val_size, random_state=seed)
    num_features = X_train.shape[1]

    print('Dataset details:')
    print(f'{X_train.shape=}, {X_val.shape = }, {X_test.shape=}')
    print(f'Positive points=> train= {y_train.sum()}, val = {y_val.sum()}, test= {y_test.sum()}')

    normalize = 'min_max' # other option 'std'
    if hasattr(dconf, 'normalize'):
        normalize = dconf.normalize
    
    if normalize == 'min_max':
        normalize_fn = MinMaxScaler()
    elif normalize == 'std':
        normalize_fn = StandardScaler()
    if normalize != False:
        X_train = normalize_fn.fit_transform(X_train)
        X_val = normalize_fn.transform(X_val)
        X_test = normalize_fn.transform(X_test)

    if model_name in ['tv', 'tv2']:
        train_data =  NumpyDataset(X=X_train, y=y_train)
        test_data = NumpyDataset(X=X_test, y= y_test)
        val_data = NumpyDataset(X=X_val, y= y_val)
    elif model_name in ['fc', 'svm']:
        train_data = (X_train, y_train)
        test_data = (X_test, y_test)
        val_data = (X_val, y_val)
 
    return train_data, val_data, test_data, num_features

def get_openmlcc18(dconf, dataset_id, seed=42, reduce_data = 0.8, verbosity = 1)-> Dataset2:
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

def get_openml_single_db(dconf, model_name, seed):
    
    val_size = 0.2
    if hasattr(dconf, 'val_size'):
        val_size = dconf.val_size
    multiply_test_size = 10
    if hasattr(dconf, 'multiply_test_size'):
        multiply_test_size = dconf.multiply_test_size
    # root_path = './data/openml/'
    # if hasattr(dconf, 'root_path'):
        # root_path = dconf.root_path
    train_size = 0.1
    if hasattr(dconf, 'train_size'):
        train_size = dconf.train_size
    if seed is not None:
        npr = np.random.RandomState(seed)
    else:
        npr = np.random

    dataset = openml.datasets.get_dataset(dconf.dataset_id) 
                                        #   cache_format='feather', 
                                        #   download_qualities=False) 
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute
    )
    # N = X.select_dtypes(include=[np.number]).values
    # C = X.select_dtypes(exclude=[np.number]).values
    # N = N if N.shape[1] > 0 else None
    # C = C if C.shape[1] > 0 else None

    X = np.array(X.values, dtype=np.float32)
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = np.array(y, dtype=np.float32)

    ## only do this for non binary dataset
    dataset_list = openml.datasets.list_datasets(output_format='dataframe')
    
    num_classes = dataset_list[dataset_list['did'] == dconf.dataset_id]['NumberOfClasses'].values[0]
    if num_classes!=2:
        p_class = npr.choice(np.unique(y))
        print(f'Total +ve points in dataste : {(y == p_class).sum()}')
        X, y = multiclass_to_binary(X, y, p_class=p_class, npr = npr)
        print(f'Dataset size after binarization: {X.shape[0]}')
    else:
        print(f'Dataset size initially: {X.shape[0]}')

    X_train, X_test , y_train, y_test = train_test_split(X, y, random_state=seed, stratify=y, shuffle=True)
    X_train, _ , y_train, _ = train_test_split(X_train, y_train, train_size=train_size, random_state=seed, stratify=y_train)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size= val_size, random_state=seed)
    num_features = X_train.shape[1]
    
    ## increasing test_data size
    X_test , y_test = increase_size(X_test, y_test, times= multiply_test_size)

    print('Dataset details:')
    print(f'{X_train.shape=}, {X_val.shape = }, {X_test.shape=}')
    print(f'Positive points=> train= {y_train.sum()}, val = {y_val.sum()}, test= {y_test.sum()}')

    normalize = 'min_max' # other option 'std'
    if hasattr(dconf, 'normalize'):
        normalize = dconf.normalize
    
    if normalize == 'min_max':
        normalize_fn = MinMaxScaler()
    elif normalize == 'std':
        normalize_fn = StandardScaler()
    if normalize != False:
        X_train = normalize_fn.fit_transform(X_train)
        X_test = normalize_fn.transform(X_test)
        X_val = normalize_fn.transform(X_val)

    if model_name in ['tv', 'tv2']:
        train_data =  NumpyDataset(X=X_train, y=y_train)
        test_data = NumpyDataset(X=X_test, y= y_test)
        val_data = NumpyDataset(X=X_val, y= y_val)
    elif model_name in ['fc', 'svm']:
        train_data = (X_train, y_train)
        test_data = (X_test, y_test)
        val_data = (X_val, y_val)

    return train_data, val_data, test_data, num_features

def multiclass_to_binary(X,y, p_class, npr, p_cls_size = None, flatten = False):
    dataset_cls = np.unique(y).tolist()
    print(f'{dataset_cls = }')
    class_idx = {key: [] for key in dataset_cls}  # class: [list of idx]
    
    for idx in range(X.shape[0]):
        class_idx[y[idx]].append(idx)

    if p_cls_size is None or p_cls_size == -1:
        p_cls_size = len(class_idx[p_class])
    else:
        p_cls_size = min(len(class_idx[p_class]), p_cls_size)
    
    positive_points = []
    negative_points = []
    for cls in dataset_cls:
        if cls != p_class:
            class_idx[cls] = npr.choice(np.array(class_idx[cls]), 
                                        p_cls_size//(len(dataset_cls)-1), 
                                        replace=False).tolist()
            negative_points += class_idx[cls]
        else:
            class_idx[cls] = npr.choice(np.array(class_idx[cls]), 
                                        p_cls_size, 
                                        replace=False).tolist()
            positive_points += class_idx[cls]

    X_new = []
    y_new = []
    for idx in positive_points:
        X_new.append(X[idx])
        y_new.append(1)
    for idx in negative_points:
        X_new.append(X[idx])
        y_new.append(0)
    permute_idx = npr.permutation(range(len(X_new)))
    X_new, y_new = np.array(X_new, dtype=np.float32)[permute_idx], np.array(y_new, dtype=np.float32)[permute_idx]

    return X_new, y_new

def increase_size(X, y, times = 1):
    Large_X = X
    Large_y = y
    for i in range(times):
        Large_X = np.append(Large_X,X, axis=0)
        Large_y = np.append(Large_y,y, axis=0)
        
    return Large_X, Large_y

def test_code():
    config = get_config('./exp0.3_tp_fc_fitting/config.yaml')
    # get_mnist(config.data)
    get_cifar10(config.data)

# test_code()
