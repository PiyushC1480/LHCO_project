import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, sys, random
import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models import resnet50, resnet18
from torch.optim import lr_scheduler
from tqdm import tqdm
import torchvision.models as models

random.seed(42)

#directory structure : 
#img \ 
#   train \
#     signal \
#     bg \
#
#   val \
#     signal \
#     bg \
#
#   test \
#     signal \
#     bg \

# Hyperparameters
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


def data_loaders(dir):
    """
    input : path to directory 
    output : train, val and test dataloaders
    """

    train_dir = os.path.join(dir, "train")  
    val_dir = os.path.join(dir, "val")
    test_dir = os.path.join(dir, "test")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=test_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    print("Size of datasets: train  : ", len(train_dataset), "val:  ",len(val_dataset), "test :  ",len(test_dataset))

    class_names= train_dataset.classes

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return  train_loader, val_loader, test_loader, class_names


# plot train loss v/s epochs and val loss v/s epochs
# def plot_loss(train_loss, val_loss):
#     """
#     input : train and val loss
#     output : plot of train and val loss
#     """

#     plt.plot(train_loss, label='train')
#     plt.plot(val_loss, label='val')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.savefig("../../output/results/loss.png")
#     plt.close()

# def plot_acc(train_acc, val_acc):
#     """
#     input : train and val accuracy
#     output : plot of train and val accuracy
#     """

#     plt.plot(train_acc, label='train')
#     plt.plot(val_acc, label='val')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.savefig("../../output/results/accuracy.png")
#     plt.close()



# Custom CNN mdoel
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



def train_model(train_loader, val_loader):

    # model = resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model = CNNBinaryClassifier()
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_acc = 0.0
    best_model_wts = model.state_dict()

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloader, desc="Processing Batches"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    print('Best val Acc: {:4f}'.format(best_acc))
    #save the model
    # torch.save(model.state_dict(), "../../output/models/model.pth")
    # plot_loss(train_loss, val_loss)
    # plot_acc(train_acc, val_acc)

    return model, best_model_wts 


def test_model(model, test_loader):
    # model.load_state_dict(torch.load("../../output/models/model.pth"))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    return correct / total



def visualize_kernels(model, layer_idx=0):
    """
    Visualizes the learned kernels of a specified convolutional layer in a CNN model.

    Parameters:
        model (torch.nn.Module): The trained CNN model.
        layer_idx (int): Index of the convolutional layer in model.conv_layers.

    Returns:
        None (Displays the kernel visualization)
    """
    # Extract the convolutional layer
    conv_layer = None
    conv_count = 0
    for layer in model.conv_layers:
        if isinstance(layer, torch.nn.Conv2d):
            if conv_count == layer_idx:
                conv_layer = layer
                break
            conv_count += 1
    
    if conv_layer is None:
        print("Invalid layer index or no Conv2d layer found.")
        return

    kernels = conv_layer.weight.data.cpu().numpy()  # Shape: (out_channels, in_channels, H, W)
    
    num_kernels = kernels.shape[0]  # Number of filters
    fig, axes = plt.subplots(1, num_kernels, figsize=(num_kernels * 2, 2))
    
    for i in range(num_kernels):
        kernel = kernels[i, 0]  # Taking the first input channel for visualization
        
        ax = axes[i] if num_kernels > 1 else axes
        ax.imshow(kernel)
        ax.set_title(f'Filter {i}')
        ax.axis('off')
    
    plt.savefig(f"../../output/CNN_out/kernel_{layer_idx}.png")


if __name__ == "__main__":

    #LOCAL FILES
    data_dir = "../../dataset/eta_phi_images/"

    # KAGGLE FILES 
    # data_dir = "/kaggle/input/trial-dataset/eta_phi_images"
    # data_dir = "/kaggle/input/dataset/img"

    train_loader, val_loader, test_loader, class_names = data_loaders(data_dir)
    model, best_model_wts = train_model(train_loader, val_loader)
    test_acc = test_model(model, test_loader)
    visualize_kernels(model,0)
    visualize_kernels(model,1)
    visualize_kernels(model,2)



    # model 
    # model = resnet50(pretrained = True)
    # #change the input layer to match the image size
    # model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # #change the last layer
    # model.fc = nn.Linear(in_features=2048, out_features=1, bias=True)
    # model.to(device)


    # # loss and optimizer
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)
    # schedulers = lr_scheduler.CosineAnnealingLR(optimizer, epochs, verbose=False)



"""

LR : 0.001 , batch size = 32 , epochs = 10 , normalization removed

Size of datasets: train  :  58690 val:   19573 test :   19570
Epoch 0/9
----------
Processing Batches: 100%|██████████| 1835/1835 [03:56<00:00,  7.78it/s]
train Loss: 0.4044 Acc: 0.9078
Processing Batches: 100%|██████████| 612/612 [01:17<00:00,  7.85it/s]
val Loss: 0.3663 Acc: 0.9449
Epoch 1/9
----------
Processing Batches: 100%|██████████| 1835/1835 [04:01<00:00,  7.61it/s]
train Loss: 0.3696 Acc: 0.9418
Processing Batches: 100%|██████████| 612/612 [01:19<00:00,  7.72it/s]
val Loss: 0.3655 Acc: 0.9457
Epoch 2/9
----------
Processing Batches: 100%|██████████| 1835/1835 [03:58<00:00,  7.68it/s]
train Loss: 0.3662 Acc: 0.9446
Processing Batches: 100%|██████████| 612/612 [01:18<00:00,  7.78it/s]
val Loss: 0.3626 Acc: 0.9484
Epoch 3/9
----------
Processing Batches: 100%|██████████| 1835/1835 [03:58<00:00,  7.69it/s]
train Loss: 0.3640 Acc: 0.9464
Processing Batches: 100%|██████████| 612/612 [01:18<00:00,  7.85it/s]
val Loss: 0.3610 Acc: 0.9503
Epoch 4/9
----------
Processing Batches: 100%|██████████| 1835/1835 [03:57<00:00,  7.72it/s]
train Loss: 0.3625 Acc: 0.9479
Processing Batches: 100%|██████████| 612/612 [01:17<00:00,  7.89it/s]
val Loss: 0.3602 Acc: 0.9511
Epoch 5/9
----------
Processing Batches: 100%|██████████| 1835/1835 [03:56<00:00,  7.75it/s]
train Loss: 0.3611 Acc: 0.9501
Processing Batches: 100%|██████████| 612/612 [01:18<00:00,  7.75it/s]
val Loss: 0.3597 Acc: 0.9519
Epoch 6/9
----------
Processing Batches: 100%|██████████| 1835/1835 [03:56<00:00,  7.75it/s]
train Loss: 0.3605 Acc: 0.9507
Processing Batches: 100%|██████████| 612/612 [01:17<00:00,  7.91it/s]
val Loss: 0.3589 Acc: 0.9518
Epoch 7/9
----------
Processing Batches: 100%|██████████| 1835/1835 [03:57<00:00,  7.74it/s]
train Loss: 0.3605 Acc: 0.9511
Processing Batches: 100%|██████████| 612/612 [01:17<00:00,  7.93it/s]
val Loss: 0.3654 Acc: 0.9464
Epoch 8/9
----------
Processing Batches: 100%|██████████| 1835/1835 [03:57<00:00,  7.72it/s]
train Loss: 0.3596 Acc: 0.9520
Processing Batches: 100%|██████████| 612/612 [01:18<00:00,  7.83it/s]
val Loss: 0.3609 Acc: 0.9493
Epoch 9/9
----------
Processing Batches: 100%|██████████| 1835/1835 [03:55<00:00,  7.79it/s]
train Loss: 0.3594 Acc: 0.9522
Processing Batches: 100%|██████████| 612/612 [01:17<00:00,  7.86it/s]
val Loss: 0.3605 Acc: 0.9514
Best val Acc: 0.951872
Accuracy of the network on the test images: 95 %



"""