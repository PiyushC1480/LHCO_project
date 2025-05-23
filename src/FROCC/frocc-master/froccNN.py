import uproot
import torch
import torch.nn as nn
import torch.optim as optim
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

import data_gen
import frocc
import dfrocc
import sparse_dfrocc
import pardfrocc
import kernels as k
import time
np.random.seed(42)

variables = ['trackno', 'thetaJ', 'LambdaJ', 'epsJ', 'tau1']

learning_rate = 1e-3  # Learning rate
batch_size = 64  # Batch size
epochs = 50  # Number of epochs


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, dropout_prob=0.5):
        super(NeuralNetwork, self).__init__()

        self.layers = nn.ModuleList()
        all_dims = [input_dim] + hidden_dims

        for i in range(len(hidden_dims)):
            self.layers.append(nn.Linear(all_dims[i], all_dims[i+1]))
            self.layers.append(nn.BatchNorm1d(all_dims[i+1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_prob))

        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


def create_df(file_name, label):
    file = uproot.open(f"../../../dataset/{file_name}.root")
    tree_name = list(file.keys())[0]
    tree = file[tree_name]
    # df = tree.arrays(library="pd")
    df = tree.arrays(variables, library="pd")
    df["label"] = label
    return df


if __name__ == "__main__":

    signal_file = "tauhadronic1L_minpt50_out"
    background_file = "hardqcd_200k_minpt50_out2"

    signal_df = create_df(signal_file, 0)
    background_df = create_df(background_file, 1)

    print(len(signal_df), len(background_df))

    df = pd.concat([signal_df, background_df], axis=0)
    df = df.sample(frac=1).reset_index(drop=True)

    X = df[variables]
    y = df["label"]

    df['label'] = y
    df_class1 = df[df['label'] == 0]
    df_class0 = df[df['label'] == 1]

    print(df_class0.shape)
    print(df_class1.shape)

    # Train: 80% of class 1
    df_train = df_class0.sample(frac=0.75, random_state=42)
    
    # Remaining 20% of class 1 for test
    df_test_1 = df_class0.drop(df_train.index)
    df_test_0 = df_class1

    df_test = pd.concat([df_test_1, df_test_0]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Final train, val, test sets
    X_train = df_train[variables]
    y_train = df_train["label"]

    X_test = df_test[variables]
    y_test = df_test["label"]

    # Print sizes to verify
    print("Train:", X_train.shape, y_train.value_counts().to_dict())
    print("Test :", X_test.shape, y_test.value_counts().to_dict())

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    # Check types to confirm
    print(type(X_train), type(y_train))


    # frocc_kernel = k.linear()
    # frocc_clf = frocc.FROCC(num_clf_dim=1000, epsilon=1000, kernel=frocc_kernel)


    # frocc_clf.fit(X_train, y_train)
    # frocc_scores = frocc_clf.decision_function(X_test)
    # frocc_roc = roc_auc_score(y_test, frocc_scores)  #0.5016041974944792 , 0.48351826782525736

    # print("Frocc scores  ::: " , frocc_scores)
    # print("Frocc ROC  ::: " ,frocc_roc)



    #dforcc model
    dfrocc_kernel = k.linear()
    dfrocc_clf = dfrocc.DFROCC(num_clf_dim=1000, epsilon=0.0001, kernel=dfrocc_kernel)
    dfrocc_clf.fit(X_train, y_train)

    #NN model
    nn_model = NeuralNetwork( input_dim=6, output_dim=2,  hidden_dims=[256, 128, 64,32], dropout_prob=0.4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(nn_model.parameters(), lr=learning_rate)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)


    train_dfrocc_scores = dfrocc_clf.decision_function(X_train)
    train_dfrocc_scores_tensor = torch.tensor(train_dfrocc_scores, dtype=torch.float32)
    X_train_combined = torch.cat((X_train_tensor, train_dfrocc_scores_tensor.unsqueeze(1)), dim=1)
    print(X_train_combined.shape)

    test_dfrocc_scores = dfrocc_clf.decision_function(X_test)
    test_dfrocc_scores_tensor = torch.tensor(test_dfrocc_scores, dtype=torch.float32)
    X_test_combined = torch.cat((X_test_tensor, test_dfrocc_scores_tensor.unsqueeze(1)), dim=1)
    print(X_test_combined.shape)
    y_test = torch.tensor(y_test.values, dtype=torch.long).numpy()

    train_dataset = torch.utils.data.TensorDataset(X_train_combined, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        nn_model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = nn_model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            # Compute accuracy for this batch
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

            loss.backward()
            optimizer.step()

        nn_model.eval()
        with torch.no_grad():
            nn_outputs = nn_model(X_test_combined)
            _, predicted = torch.max(nn_outputs, 1)
            # Convert predicted to numpy
            predicted = predicted.numpy()
            # Convert y_test to numpy
            
            # Calculate the accuracy
            accuracy = (predicted == y_test).sum().item() / len(y_test)
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss},train Accuracy: {epoch_acc}, test_acc : {accuracy}")


    
        

    # nn_scores = nn_outputs[:, 1].numpy()  # Get the scores for the positive class
    # nn_roc = roc_auc_score(y_test, nn_scores)
    # print("NN ROC  ::: " ,nn_roc)

    nn_scores = nn_outputs[:, 1].numpy()  # Get the scores for the positive class
    nn_final_out = [1 if score > 0.5 else 0 for score in nn_scores]
    nn_roc = roc_auc_score(y_test, nn_final_out)
    nn_acc = accuracy_score(y_test, nn_final_out)
    nn_f1 = f1_score(y_test, nn_final_out)
    print("NN ROC  ::: " ,nn_roc)
    print("NN ACC  ::: " ,nn_acc)
    print("NN F1  ::: " ,nn_f1)







    # dfrocc_scores = dfrocc_clf.decision_function(X_test)
    # # y_test = 1 - y_test
    # dfrocc_roc = roc_auc_score(y_test, dfrocc_scores)  

    # print("DFrocc scores  ::: " ,dfrocc_scores , y_test)
    # print("DFrocc roc  ::: " ,dfrocc_roc)
    # print(dfrocc_scores[:20])
    # print(y_test[:20])



    # pardfrocc_kernel = k.linear()
    # pardfrocc_clf = pardfrocc.ParDFROCC(num_clf_dim=1000, epsilon=1000, kernel=pardfrocc_kernel)


    # pardfrocc_clf.fit(X_train, y_train)
    # pardfrocc_scores = pardfrocc_clf.decision_function(X_test)
    # pardfrocc_roc = roc_auc_score(y_test, pardfrocc_scores)  #0.5016041974944792

    # print("ParDFrocc scores  ::: " ,pardfrocc_scores)
    # print("ParDFrocc scores  ::: " ,pardfrocc_roc)





    

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



    # x, y, _, _, xtest, ytest = data_gen.himoon(
    #     n_samples=1000, n_dims=1000
    # )

    # print(f"Type of x : {type(x)} ")
    # print(f"Type of y : {type(y)} ")
    # print(f"Type of x : {type(x)} , Shape of x is  : {x.shape}")
