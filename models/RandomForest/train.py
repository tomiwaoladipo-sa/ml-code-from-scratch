# This code trains a decision tree and a random forest on the breast cancer dataset
# It then compares the performance of the custom decision tree and random forest with the sklearn implementation

import numpy as np
import pandas as pd
from DecisionTree import DecisionTree
from RandomForest import RandomForest
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


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

train_features, test_features, train_target, test_target = train_test_split(
    arr_features, arr_target, test_size=0.95, random_state=100
)


def test_models(train_features, test_features, train_target, test_target):

    # Build the decision tree
    dt = DecisionTree()
    dt_custom_tree = dt.fit(
        train_features, train_target, max_depth=10, num_features=train_features.shape[1]
    )

    # Build the random forest
    rf = RandomForest()
    rf_custom_tree = rf.fit(
        train_features, train_target, max_depth=5, num_features=10, max_estimators=25
    )

    # Create a decision tree classifier
    clf = RandomForestClassifier(
        criterion="entropy", max_depth=5, max_features=10, n_estimators=20
    )

    # Train the classifier
    clf.fit(train_features, train_target)

    # # Print the depth of the decision tree
    # print("depth: ", count_depth(custom_tree))

    # # Make predictions using the decision tree & random forest
    dt_train_predictions = dt.predict(train_features)
    dt_test_predictions = dt.predict(test_features)

    rf_train_predictions = rf.predict(train_features)
    rf_test_predictions = rf.predict(test_features)

    sk_rf_train_predictions = clf.predict(train_features)
    sk_rf_test_predictions = clf.predict(test_features)

    return (
        dt_train_predictions,
        dt_test_predictions,
        rf_train_predictions,
        rf_test_predictions,
        sk_rf_train_predictions,
        sk_rf_test_predictions,
        rf_custom_tree,
        dt_custom_tree,
    )


iterations = 10
rf_train_miscalc = []
rf_test_miscalc = []
dt_train_miscalc = []
dt_test_miscalc = []
sk_rf_train_miscalc = []
sk_rf_test_miscalc = []
dt_custom_tree, rf_custom_tree = None, None
for i in range(iterations):
    (
        dt_train_predictions,
        dt_test_predictions,
        rf_train_predictions,
        rf_test_predictions,
        sk_rf_train_predictions,
        sk_rf_test_predictions,
        rf_custom_tree,
        dt_custom_tree,
    ) = test_models(train_features, test_features, train_target, test_target)

    rf_train_miscalc.append(
        len(train_target) - sum(train_target == rf_train_predictions)
    )
    rf_test_miscalc.append(len(test_target) - sum(test_target == rf_test_predictions))
    dt_train_miscalc.append(
        len(train_target) - sum(train_target == dt_train_predictions)
    )
    dt_test_miscalc.append(len(test_target) - sum(test_target == dt_test_predictions))

    sk_rf_train_miscalc.append(
        len(train_target) - sum(train_target == sk_rf_train_predictions)
    )
    sk_rf_test_miscalc.append(
        len(test_target) - sum(test_target == sk_rf_test_predictions)
    )

# print the predictions
# print(predictions)

# # Print the number of misclassifications
print(
    f"Iterations: {iterations}\n",
    "Random Forest: \n",
    "\t train misclassifications: ",
    np.mean(rf_train_miscalc),
    round(np.mean(rf_train_miscalc) / len(train_target), 2),
    "\n \t test misclassifications: ",
    np.mean(rf_test_miscalc),
    round(np.mean(rf_test_miscalc) / len(test_target), 2),
    "\n Decision Tree: \n",
    "\t train misclassifications: ",
    np.mean(dt_train_miscalc),
    round(np.mean(dt_train_miscalc) / len(train_target), 2),
    "\n \t test misclassifications: ",
    np.mean(dt_test_miscalc),
    round(np.mean(dt_test_miscalc) / len(test_target), 2),
    "\n Random Forest (sklearn): \n",
    "\t train misclassifications: ",
    np.mean(sk_rf_train_miscalc),
    round(np.mean(sk_rf_train_miscalc) / len(train_target), 2),
    "\n \t test misclassifications: ",
    np.mean(sk_rf_test_miscalc),
    round(np.mean(sk_rf_test_miscalc) / len(test_target), 2),
)

# print the number of estimators
print(len(rf_custom_tree))

# print first left branch for the first two estimators
# print(custom_tree[0]["left"], "\n 2: \n", custom_tree[1]["left"])
print(train_features.shape)

print(dt_custom_tree)
