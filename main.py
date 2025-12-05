from __future__ import annotations
from pathlib import Path

import torch
from torch import nn
from torch.cuda import device
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

import os
import pandas as pd
import numpy as np
from fontTools.varLib.avar.plan import WEIGHTS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import ConvolutionalNeuralNetworks
from database import FeatureConfig, build_train_dataframe, build_test_dataframe, \
    save_database_artifacts
#from Utils import train_svm_rbf, train_random_forest, train_gaussian_nb, train_gradient_boost
#from LogisticRegressionMultiClass import LogisticRegressionMultiClass


#Figured out way to have file path work
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
SPEC_ROOT = PROC / "spectrograms"

"""************************************************************* CNN Parameters ***************************************************************************"""
CNN_TRAINING_DATA_PATH = os.path.join(SPEC_ROOT, "train")
IMAGE_SIZE = 128                                                #Don't change this one. The images need to be that size to fit into the CNN

BATCH_SIZE = 32
EPOCHS = 40
BASE_FILTERS=32
HIDDEN_FILTERS=BASE_FILTERS*2
LR=0.001
WEIGHT_DECAY=0.0001
VALIDATION_SPLIT=0.2
DROPOUT=0.1
PADDING=1
KERNEL_SIZE=3
"""********************************************************************************************************************************************************"""
#Toggles so we don't redo work every run
BUILD_FEATURE_DATABASE = False
GENERATE_SPECTROGRAMS = False   # flip to False when you don't want to regenerate



"""**************************************************************** Parameters ***************************************************************************"""
mfcc = True
fcc_delta=False
chroma=True
spectral_contrast=True
zcr=True
spectral_centroid=True
spectral_bandwidth=True
spectral_rolloff=True
rms=True
tempo=False
n_mfcc=20
aggregation="mean_std"

#How far we move
logistRegressStepSize = 0.1
#How many iterations of training
logistRegressEpochs = 300
"""*******************************************************************************************************************************************************"""


#Change these paramters to try a fuckton of different things
def build_database():
    """
    :return:
    """
    print("Configuring features...")
    cfg = FeatureConfig(
        mfcc=False,
        mfcc_delta=True,
        chroma=False,
        spectral_contrast=False,
        zcr=False,
        spectral_centroid=False,
        spectral_bandwidth=False,
        spectral_rolloff=True,
        rms=True, tempo=False,
        n_mfcc=5,
        aggregation="mean_std",
    )
    print("Features configured, building training dataframe")
    df_tr = build_train_dataframe(str(RAW / "train"), cfg)
    print("Training dataframe built, building test dataframe")
    df_te = build_test_dataframe(str(RAW / "test"), cfg)
    print("Test dataframe built, saving database artifacts")
    save_database_artifacts(df_tr, df_te, str(PROC))
    print("Done")
    return df_tr, df_te

def build_spectrograms():
    """
    Generate spectrogram PNGs for all train and test audio files
    """
    from database import generate_spectrograms_for_train, generate_spectrograms_for_test

    print("Generating spectrograms for training data")
    generate_spectrograms_for_train(str(RAW / "train"), str(SPEC_ROOT))
    print("Generating spectrograms for test data")
    generate_spectrograms_for_test(str(RAW / "test"), str(SPEC_ROOT))
    print("Done generating spectrograms.")

if __name__ == "__main__":
    if BUILD_FEATURE_DATABASE:
        train_df, test_df = build_database()

    if GENERATE_SPECTROGRAMS:
        build_spectrograms()


#Reshaping the generated sptrograms to match the CNN input
train_transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.RandomHorizontalFlip(), transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
val_transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

cnn_training_dataset = datasets.ImageFolder(root=CNN_TRAINING_DATA_PATH, transform=train_transform)

numClasses = len(cnn_training_dataset.classes)
print("We've got " + str(numClasses) + " classes")

numSamples = len(cnn_training_dataset)
numVal = int(numSamples * VALIDATION_SPLIT)
numTrain = numSamples - numVal

print("Splitting the dataset into training and validation")
cnn_train_dataset, cnn_val_dataset = random_split(cnn_training_dataset, [numTrain, numVal], generator=torch.Generator().manual_seed(67))


cnn_val_dataset.dataset.transform = val_transform

print("Intitializing loaders")
cnn_training_loader = DataLoader(cnn_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
cnn_val_loader = DataLoader(cnn_val_dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Setting up the torch device (will be CPU) and the model")
device = torch.device("cpu")
model = ConvolutionalNeuralNetworks.CNN(numClasses=numClasses, in_channels=3, baseFilters=BASE_FILTERS, hiddenFilters=HIDDEN_FILTERS, dropout=DROPOUT, kernelSize=KERNEL_SIZE, paddingInput=PADDING)

criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

print("Training the model " + str(EPOCHS) + " epochs")
best_acc = 0.0
for epoch in range(EPOCHS):
    print("Epoch: " + str(epoch) + " of " + str(EPOCHS))
    train_loss, train_acc = model.trainOnePass(cnn_training_loader, optimizer, criteria, device=device)
    val_loss, val_acc = model.predict(cnn_val_loader, criterion=criteria, device=device)

    if val_acc > best_acc:
        best_acc = val_acc

print("Best accuracy: " + str(best_acc))
