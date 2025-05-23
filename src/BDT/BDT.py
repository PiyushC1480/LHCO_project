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
import sys

"""
Plot the histogram plot for the given variables for signal and background
"""
variables = ["trackno","LambdaJ", "ecfdr2", "tau31"]
bins = [ np.linspace(0, 15, 15) ,  np.linspace(-3, 0, 31), np.linspace(0,0.045,46),np.linspace(0,0.9,45) ]

def create_df(file_name, label):
    file = uproot.open(f"../dataset/{file_name}.root")
    tree_name = list(file.keys())[0]
    tree = file[tree_name]
    df = tree.arrays(library="pd")
    # df = tree.arrays(variables, library="pd")
    df["label"] = label
    return df


def plot_variable(variable_idx, signal_df, background_df):
    #normalize the histograms bins
    variable_name = variables[variable_idx]
    b = bins[variable_idx]
    plt.hist(signal_df[variable_name], bins=b, histtype='step', linestyle='-' ,color='blue', label="tau", density=True)
    plt.hist(background_df[variable_name], bins=b, color='red', histtype='step', linestyle=':', label="QCD", density=True)
    plt.xlabel(variable_name)
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"../output/new_bdt_plots/{variable_name}.png")
    print(f"Plotted {variable_name}")
    plt.close()
    return



if __name__ == "__main__":

    # signal_file = "tauhadronic1L_minpt50_out"
    signal_file = "tauhadronic_out"
    background_file = "fixed_bg"

    signal_df = create_df(signal_file, 1)
    background_df = create_df(background_file, 0)

    # for i in range(0,4):
    # plot_variable(2, signal_df, background_df)
        # break
    # print(len(signal_df), len(background_df))
    