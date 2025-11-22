from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from database import FeatureConfig, build_train_dataframe, build_test_dataframe, \
    save_database_artifacts
#from Utils import train_svm_rbf, train_random_forest, train_gaussian_nb, train_gradient_boost
#from LogisticRegressionMultiClass import LogisticRegressionMultiClass


#Figured out way to have file path work
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"


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


#Change these paramters to try a fuckton of different things,
def build_database():
    """
    :return:
    """
    print("Configuring features...")
    cfg = FeatureConfig(
        mfcc=True,
        mfcc_delta=False,
        chroma=True,
        spectral_contrast=True,
        zcr=True,
        spectral_centroid=True,
        spectral_bandwidth=True,
        spectral_rolloff=True,
        rms=True, tempo=False,
        n_mfcc=20,
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