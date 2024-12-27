import numpy as np
import pandas as pd
from DecisionTree import DecisionTree
from sklearn.datasets import load_breast_cancer


# Function to count the depth of a dictionary (used to measure the depth of the decision tree)
def count_depth(d):
    if not isinstance(d, dict) or not d:
        return 0
    return 1 + max(count_depth(v) for v in d.values())


# Load the breast cancer dataset
breast_cancer = load_breast_cancer(as_frame=True)
df_breast_cancer = breast_cancer.data

# Convert the features dataframe to a numpy array
arr_features = np.array(df_breast_cancer.copy())

# Convert the target series to a numpy array
arr_target = np.array(breast_cancer.target.copy())

# Build the decision tree
dt = DecisionTree()
custom_tree = dt.fit(arr_features, arr_target, max_depth=1, num_features=1)

# Print the depth of the decision tree
print("depth: ", count_depth(custom_tree))

# Make predictions using the decision tree
predictions = dt.predict(arr_features)

# Print the number of misclassifications
print("misclassifications: ", len(arr_target) - sum(predictions == arr_target))
# print(custom_tree)
