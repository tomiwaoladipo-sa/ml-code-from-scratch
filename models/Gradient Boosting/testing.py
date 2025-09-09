# I found addings probabilities using sigmoid is wrong

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from GradientBoosting_Rough import GradientBoosting

# from RandomForest import RandomForest
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split


def count_depth(d):
    if not isinstance(d, dict) or not d:
        return 0
    return 1 + max(count_depth(v) for v in d.values())


# Load the breast cancer dataset
breast_cancer = load_breast_cancer(as_frame=True)
df_breast_cancer = breast_cancer.data

df_breast_cancer = df_breast_cancer[["worst radius"]].copy()

# Convert the features dataframe to a numpy array
arr_features = np.array(df_breast_cancer.copy())

# Convert the target series to a numpy array
arr_target = np.array(breast_cancer.target.copy())


# join arr_features and arr_target
arr_features = np.c_[arr_features, arr_target]

# mean of arr_features when arr_target is 0 and 1
mean_0 = arr_features[arr_features[:, 1] == 0][:, 0].mean()
mean_1 = arr_features[arr_features[:, 1] == 1][:, 0].mean()

print(mean_0, mean_1)
