import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import pdb
import numpy as np
import typing as ty
import uproot
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
from model import *
np.random.seed(42)

variables = ['trackno', 'thetaJ', 'LambdaJ', 'epsJ', 'tau1']

def create_df(file_name, label):
    file = uproot.open(f"../../dataset/{file_name}.root")
    tree_name = list(file.keys())[0]
    tree = file[tree_name]
    # df = tree.arrays(library="pd")
    df = tree.arrays(variables, library="pd")
    df["label"] = label
    return df




if __name__ == "__main__":

    signal_file = "tauhadronic1L_minpt50_out"
    background_file = "hardqcd_200k_minpt50_out2"

    signal_df = create_df(signal_file, 1)
    background_df = create_df(background_file, 0)

    print(len(signal_df), len(background_df))

    df = pd.concat([signal_df, background_df], axis=0)
    df = df.sample(frac=1).reset_index(drop=True)

    X = df[variables]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_tes_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    #print shapes
    print(f"X_train_tensor shape: {X_train_tensor.shape}")
    print(f"y_train_tensor shape: {y_train_tensor.shape}")
    print(f"X_test_tensor shape: {X_test_tensor.shape}")
    print(f"y_test_tensor shape: {y_tes_tensor.shape}")



    #model
    model = TV7(
        d_in=X_train_tensor.shape[1],
    )

    
    # forward pass
    model.init_phase2(X_train_tensor, None, y_train_tensor, 10)

    #loss funciton and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    #cerate a dataloader for train and test
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_tes_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    #train
    num_epochs = 10
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs, None)
            loss = criterion(outputs, labels) + model.regularizer()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}")


    #print train score
    train_scores = model(X_train_tensor, None)
    print(f"Train scores: {train_scores}")
    print(train_scores[:10])
    print(y_train[:10])
