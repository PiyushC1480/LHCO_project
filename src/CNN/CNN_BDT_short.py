import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, sys, random
import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50, resnet18
from torch.optim import lr_scheduler
from tqdm import tqdm
import torchvision.models as models
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import *
from sklearn.metrics import *
import torchvision.datasets as datasets
import csv
import os


"""
CNN + BDT

take the pretrained CNN model and train BDT now.
"""
# ------------------ HYPER PARAMETERS -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 10
batch_size = 32
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    # transforms.Normalize(mean, std),
])

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    # transforms.Normalize(mean, std),
])


# ----------------- CUSTOM FUNCTIONS -----------------
def split_root_file(root_file, train_ratio=0.6, val_ratio=0.2):
    # Read the ROOT file into a DataFrame
    file = uproot.open(f"{root_file}")
    tree_name = list(file.keys())[0]
    tree = file[tree_name]
    df = tree.arrays(library="pd")
    # Shuffle the data
    # df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    # Compute split indices
    train_end = int(len(df) * train_ratio)
    val_end = train_end + int(len(df) * val_ratio)
    
    # Split the DataFrame
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    return train_df, val_df, test_df

def store_root_file(root_file,name):
    train_df, val_df, test_df = split_root_file(root_file)

    # Save the DataFrames to CSV files
    train_df.to_csv(f"../../dataset/pts_order/train/{name}.csv", index=False)
    val_df.to_csv(f"../../dataset/pts_order/val/{name}.csv", index=False)
    test_df.to_csv(f"../../dataset/pts_order/test/{name}.csv", index=False)

    print("stored train, val, test files in pts_order folder")
    return

# ------------------- Data Set -------------------
class CustomDataset(Dataset):
    def __init__(self, csv_folder, img_folder, transform=None):
        #signal information
        sig_csv_file = os.path.join(csv_folder,"signal.csv")
        self.sig_data = pd.read_csv(sig_csv_file)
        self.sig_data = self.sig_data.drop(columns=['normalized_ecf1', 'normalized_ecf2', 'normalized_ecf3'])

        self.sig_img_folder = os.path.join(img_folder, "signal")

        print(len(self.sig_data))

        #bg information
        bg_csv_file = os.path.join(csv_folder,"bg.csv")
        self.bg_data = pd.read_csv(bg_csv_file)
        self.bg_img_folder = os.path.join(img_folder, "bg")
        print(len(self.bg_data))
        self.transform = transform

        self._data_list =[]
        
        for idx in range(len(self.sig_data)):
            # the name og the i_th image is i.png, so get the ith image from the folder
            img_path = os.path.join(self.sig_img_folder, str(idx) + ".png")
            img = datasets.folder.default_loader(img_path)
            # get the ith row from the csv file
            row = self.sig_data.iloc[idx]
            self._data_list.append((img, row, 1))

        for idx in range(len(self.bg_data)):
            # the name og the i_th image is i.png, so get the ith image from the folder
            img_path = os.path.join(self.bg_img_folder, str(idx) + ".png")
            img = datasets.folder.default_loader(img_path)
            # get the ith row from the csv file
            row = self.bg_data.iloc[idx]
            self._data_list.append((img, row, 0))

        #shuffle the data
        random.shuffle(self._data_list)

    def __len__(self):
        return len(self.sig_data)
    
    def __getitem__(self, idx):
        img, row, label = self._data_list[idx]
        if self.transform:
            img = self.transform(img)
        # convert the row to a tensor
        row = torch.tensor(row.values, dtype=torch.float32)
        return img, row, label
    


# ------------------- CNN Model -------------------
class CNNBinaryClassifier(nn.Module):
    def __init__(self):
        super(CNNBinaryClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(16 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x
    


def append_cnn_output(cnn_model, data_loader, output_file):
    # Remove the existing file if it exists
    if os.path.isfile(output_file):
        os.remove(output_file)

    with open(output_file, mode='w', newline='') as file:  # 'w' mode to create a new file
        writer = csv.writer(file)
        
        # Writing the header row
        first_batch = next(iter(data_loader))
        row_keys = [f'row_{i}' for i in range(len(first_batch[1][0]))]  # Creating column names for row data
        writer.writerow(row_keys + ['cnn_output', 'label'])
        i =0 
        for imgs, rows, labels in tqdm(data_loader):
            imgs = imgs.to(device)
            cnn_output = cnn_model(imgs).cpu().detach().numpy()  # Get CNN output
            #select the index of the max element
            cnn_output = np.argmax(cnn_output, axis=1)  # Get the index of the max element
            cnn_output = torch.tensor(cnn_output, dtype=torch.float32)  # Convert to tensor
    
            
            for row, output, label in zip(rows, cnn_output, labels):
                # Convert row to numpy array
                row = row.numpy()
                # Write the row data and CNN output to the CSV file
                writer.writerow(list(row) + [output.item(), label.item()])

                
    

    print(f"File created: {output_file}")

    
            
if __name__=="__main__":

    # sig_root = "../../dataset/tauhadronic1L_minpt50_out.root"
    # bg_root = "../../dataset/hardqcd_200k_minpt50_out2.root"

    # store_root_file(sig_root,"signal")
    # store_root_file(bg_root, "bg")

    # train_imgs=  "/../input/trial-dataset/eta_phi_images/train"
    # val_imgs=  "/../input/trial-dataset/eta_phi_images/val"
    # test_imgs=  "/../input/trial-dataset/eta_phi_images/test"

    train_imgs=  "/../../dataset/all_imgs_order/train"
    val_imgs=  "/../../dataset/all_imgs_order/val"
    test_imgs=  "/../../dataset/all_imgs_order/test"

    train_csv = "/../../dataset/pts_order/train"
    val_csv = "/../../dataset/pts_order/val"
    test_csv = "/../../dataset/pts_order/test"

    train_ds=  CustomDataset(train_csv, train_imgs, transform=train_transform)
    # print(train_ds[0])
    val_ds=  CustomDataset(val_csv, val_imgs, transform=test_transform)
    test_ds = CustomDataset(test_csv,test_imgs, transform=test_transform)  

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    cnn_model = CNNBinaryClassifier()
    cnn_model.load_state_dict(torch.load("../../output/trained_CNN/best_model.pth", weights_only=True))
    cnn_model.eval()
    cnn_model.to(device)

    # append the cnn output to the csv file
    append_cnn_output(cnn_model, train_loader, "../../dataset/train_cnn_output.csv")
    append_cnn_output(cnn_model, val_loader, "../../dataset/val_cnn_output.csv")
    append_cnn_output(cnn_model, test_loader, "../../dataset/test_cnn_output.csv")

        