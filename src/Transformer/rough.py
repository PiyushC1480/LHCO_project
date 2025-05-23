import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#create a parquet file from the root file
def write_parquet(file_name):
    file  = uproot.open(f"../../dataset/JetClass/{file_name}.root")
    print(file.keys())
    tree_name = list(file.keys())[0]
    tree = file[tree_name]
    df = tree.arrays(library="pd")
    df.to_parquet(f"../../dataset/JetClass/converted/converted_{file_name}.parquet", index=False)
    print(f"saved {file_name}.parquet")
    return 


# Create a CSV file from the ROOT file
def write_csv(file_name):
    file = uproot.open(f"../../dataset/JetClass/{file_name}.root")
    print(file.keys())
    tree_name = list(file.keys())[0]
    tree = file[tree_name]
    df = tree.arrays(library="pd")
    df.to_csv(f"../../dataset/JetClass/converted/converted_{file_name}.csv", index=False)
    print(f"saved {file_name}.csv")
    return



if __name__ == "__main__":
    fn = "JetClass_example_100k"

    write_csv(fn)