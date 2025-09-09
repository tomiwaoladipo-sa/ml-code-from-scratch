import numpy as np
import pandas as pd
from LogisticRegression import LogisticRegression
from sklearn.datasets import load_breast_cancer

# Load the breast cancer dataset
breast_cancer = load_breast_cancer(as_frame=True)
df_breast_cancer = breast_cancer.data

# Convert the features dataframe to a numpy array
X = np.array(df_breast_cancer.copy())

# Convert the target series to a numpy array
y = np.array(breast_cancer.target.copy())

X = np.c_[np.ones(X.shape[0]), X]

# ensure y numy array is in the right structure
y_reshaped = y.reshape(1, y.shape[0])

print(y_reshaped.shape)
predictions = y_reshaped * 0.99
n = y_reshaped.shape[1]
# print((y_reshaped - predictions) @ X)
print((-1 / n * ((y_reshaped - predictions) @ X)).shape)
