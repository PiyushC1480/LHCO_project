import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

import data_gen
import frocc
import dfrocc
import sparse_dfrocc
import pardfrocc
import kernels as k
import time
np.random.seed(42)

variables = ['trackno', 'thetaJ', 'LambdaJ', 'epsJ', 'tau1']

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



    dfrocc_kernel = k.rbf()
    dfrocc_clf = dfrocc.DFROCC(num_clf_dim=1500, epsilon=0.0001, kernel=dfrocc_kernel)


    dfrocc_clf.fit(X_train, y_train)
    dfrocc_scores = dfrocc_clf.decision_function(X_test)
    dfrocc_final_pred = [1 if dfrocc_scores[i] > 0.9 else 0 for i in range(len(dfrocc_scores))]
    # y_test = 1 - y_test
    dfrocc_roc = roc_auc_score(y_test, dfrocc_final_pred) 
    dfrocc_f1 = f1_score(y_test, dfrocc_final_pred)  
    dfrocc_accuracy = accuracy_score(y_test, dfrocc_final_pred)  #0.5016041974944792 , 0.48351826782525736

    print("DFrocc scores  ::: " ,dfrocc_scores , y_test)
    print("DFrocc accuracy  ::: " ,dfrocc_accuracy)
    print("DFrocc roc  ::: " ,dfrocc_roc)
    print("DFrocc f1  ::: " ,dfrocc_f1)
    print(dfrocc_scores[:20])
    print(y_test[:20])



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
