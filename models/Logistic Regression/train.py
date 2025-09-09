import numpy as np
import pandas as pd
import sklearn.linear_model as sklm
from LogisticRegression import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# Load the breast cancer dataset
breast_cancer = load_breast_cancer(as_frame=True)
df_breast_cancer = breast_cancer.data

columns = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    # 'mean area',
    # 'mean smoothness',
    # 'mean compactness',
    # 'mean concavity',
    # 'mean concave points',
    # 'mean symmetry',
    # 'mean fractal dimension',
    # 'radius error',
    # 'texture error',
    # 'perimeter error',
    # 'area error',
    # 'smoothness error',
    # 'compactness error',
    # 'concavity error',
    # 'concave points error',
    # 'symmetry error',
    # 'fractal dimension error',
    # 'worst radius',
    # 'worst texture',
    # 'worst perimeter',
    # 'worst area',
    # 'worst smoothness',
    # 'worst compactness',
    # 'worst concavity',
    # 'worst concave points',
    # 'worst symmetry',
    # 'worst fractal dimension'
]  # Convert the features dataframe to a numpy array
arr_features = np.array(df_breast_cancer[columns].copy())


# Convert the target series to a numpy array
arr_target = np.array(breast_cancer.target.copy())

scaler = StandardScaler()
arr_features_scaled = scaler.fit_transform(arr_features)
# Build the decision tree
lr = LogisticRegression(threshold=0.0001, learning_rate=30)
lr.fit(arr_features_scaled, arr_target)

lr_weights = lr.coefficients[0][1:]
lr_bias = lr.coefficients[0][0]

print("weights: ", lr_weights, "\n", "bias: ", lr_bias)

arr_features_scaled_w_bias = np.c_[
    np.ones(arr_features_scaled.shape[0]), arr_features_scaled
]

print(
    "accuracy: ",
    np.sum(
        np.where(
            1 / (1 + np.exp(-np.dot(lr.coefficients, arr_features_scaled_w_bias.T)))
            > 0.5,
            1,
            0,
        )
        == arr_target
    )
    / len(arr_target),
    np.sum(
        np.where(
            1 / (1 + np.exp(-np.dot(lr.coefficients, arr_features_scaled_w_bias.T)))
            > 0.5,
            1,
            0,
        )
        == arr_target
    ),
)

sk_lr = sklm.LogisticRegression()
sk_lr.fit(arr_features_scaled, arr_target)

sk_weights = sk_lr.coef_
sk_bias = sk_lr.intercept_
sk_coef = np.c_[sk_bias, sk_weights]
print(
    "sk accuracy: ",
    np.sum(
        np.where(
            1 / (1 + np.exp(-np.dot(sk_coef, arr_features_scaled_w_bias.T))) > 0.5,
            1,
            0,
        )
        == arr_target
    )
    / len(arr_target),
    np.sum(
        np.where(
            1 / (1 + np.exp(-np.dot(sk_coef, arr_features_scaled_w_bias.T))) > 0.5,
            1,
            0,
        )
        == arr_target
    ),
)

print("\n", "sk weights: ", sk_weights, "\n", "sk bias: ", sk_bias)

print(
    "is weights difference significant: ",
    abs(lr_weights - sk_weights) / abs(sk_weights) * 100 > 5,
)
print(
    "is bias difference significant: ", abs(lr_bias - sk_bias) / abs(sk_bias) * 100 > 5
)

# Make predictions using the decision tree
# predictions = lr.predict(arr_features)

# Print the number of misclassifications
# print("misclassifications: ", len(arr_target) - sum(predictions == arr_target))
# print(custom_tree)
