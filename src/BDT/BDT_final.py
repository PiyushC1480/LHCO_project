import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
import sys, random

sys.stdout = open("output_new2.txt", "w")


"""
Designing boosted decision tree for the classification of tau hadronic decay and hard QCD jets

variable to use :  theta_J and N_T initially to segment phase space and then use lambda_J, r_2 and tau_31 to analyze the subspaces using multi-variate analysis
files given  : signal (tauhadronic_out.root) and background (hardqcd_200k_minpt50_out.root)
"""
variables = ['trackno', 'thetaJ', 'LambdaJ', 'epsJ', 'tau1']
# ['trackno', 'thetaJ', 'LambdaJ', 'ecrf2']


def create_df(file_name, label):
    file = uproot.open(f"../dataset/{file_name}.root")
    tree_name = list(file.keys())[0]
    tree = file[tree_name]
    # df = tree.arrays(library="pd")
    df = tree.arrays(variables, library="pd")
    df["label"] = label
    return df


#create a parquet file from the root file
def write_parquet(file_name):
    file  = uproot.open(f"../dataset/{file_name}.root")
    print(file.keys())
    tree_name = list(file.keys())[0]
    tree = file[tree_name]
    df = tree.arrays(library="pd")
    df.to_parquet(f"../dataset/converted/converted_{file_name}.parquet", index=False)
    print(f"saved {file_name}.parquet")
    return 


if __name__ == "__main__":

    signal_file = "tauhadronic1L_minpt50_out"
    background_file = "hardqcd_200k_minpt50_out2"

    # write_parquet(signal_file)
    # write_parquet(background_file)

    signal_df = create_df(signal_file, 1)
    background_df = create_df(background_file, 0)

    print(len(signal_df), len(background_df))

    # write_parquet(signal_file)
    # write_parquet(background_file)

    df = pd.concat([signal_df, background_df], axis=0)
    df = df.sample(frac=1).reset_index(drop=True)

    X = df[variables]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    signal_test_idx = y_test[y_test == 1].index[0]  # Get one signal sample index
    background_test_idx = y_test[y_test == 0].index  # Get all background sample indices
    test_idx = [signal_test_idx] + list(background_test_idx)
    random.shuffle(test_idx)
    X_test = X_test.loc[test_idx]
    y_test = y_test.loc[test_idx]


    # Gradient Boosting Classifier
    # clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, max_depth=3, random_state=42)
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    # print(f"Accuracy GB: {accuracy_score(y_test, y_pred)}")
    # print(f"ROC AUC GB: {roc_auc_score(y_test, y_pred)}")
    # print(f"Confusion Matrix GB: \n{confusion_matrix(y_test, y_pred)}")

    # clf = GradientBoostingClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        "subsample": [0.7, 0.8, 1.0]
    }
    # # Perform Grid Search   
    # grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    # grid_search.fit(X_train, y_train)

    # # Best parameters and best model
    # print(f"Best Parameters: {grid_search.best_params_}")
    # best_model = grid_search.best_estimator_

    # y_pred = best_model.predict(X_test)
    # print(f"Accuracy GB: {accuracy_score(y_test, y_pred)}")
    # print(f"ROC AUC GB: {roc_auc_score(y_test, y_pred)}")
    # print(f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}")
    
    # print(f"Precision (Macro): {precision_score(y_test, y_pred, average='macro')}")
    # print(f"Recall (Macro): {recall_score(y_test, y_pred, average='macro')}")
    # print(f"F1-score (Macro): {f1_score(y_test, y_pred, average='macro')}")

    # print(f"Precision (Micro): {precision_score(y_test, y_pred, average='micro')}")
    # print(f"Recall (Micro): {recall_score(y_test, y_pred, average='micro')}")
    # print(f"F1-score (Micro): {f1_score(y_test, y_pred, average='micro')}")

    # XGBoost Classifier
    xgb_clf = xgb.XGBClassifier(eval_metric="logloss")
    grid_search2 = GridSearchCV(xgb_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

    grid_search2.fit(X_train, y_train)

    # Best parameters and best model
    print(f"Best Parameters: {grid_search2.best_params_}")
    best_model = grid_search2.best_estimator_

    y_pred = best_model.predict(X_test)
    print(f"Accuracy XGB: {accuracy_score(y_test, y_pred)}")
    print(f"ROC AUC XGB: {roc_auc_score(y_test, y_pred)}")
    print(f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}")

    print(f"Precision (Macro): {precision_score(y_test, y_pred, average='macro')}")
    print(f"Recall (Macro): {recall_score(y_test, y_pred, average='macro')}")
    print(f"F1-score (Macro): {f1_score(y_test, y_pred, average='macro')}")

    print(f"Precision (Micro): {precision_score(y_test, y_pred, average='micro')}")
    print(f"Recall (Micro): {recall_score(y_test, y_pred, average='micro')}")
    print(f"F1-score (Micro): {f1_score(y_test, y_pred, average='micro')}")

    # xgb_clf.fit(X_train, y_train)
    # y_pred = xgb_clf.predict(X_test)
    # print(f"Accuracy XGB: {accuracy_score(y_test, y_pred)}")
    # print(f"ROC AUC XGB: {roc_auc_score(y_test, y_pred)}")
    # print(f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}")

    sys.stdout.close()