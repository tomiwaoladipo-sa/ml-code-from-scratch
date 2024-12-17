import math

import numpy as np


def thresholds(arr):
    """
    Calculate the thresholds for an array of values.

    This function takes an array of values, sorts them, and then calculates the midpoints
    between each pair of consecutive unique values. These midpoints can be used as
    thresholds for splitting the data in decision tree algorithms.

    Parameters:
        arr (array-like): The input array of values.

    Returns:
        numpy.ndarray: An array of threshold values.
    """
    # Sort the unique values in the array and calculate midpoints between consecutive values
    arr_sorted = np.sort(np.unique(arr))
    beginning_arr = arr_sorted[0:-1]
    end_arr = arr_sorted[1:]
    return (beginning_arr + end_arr) / 2


def weights(parent_array, child_array):
    """
    Calculate the weight of a child array relative to its parent array.

    The weight is defined as the ratio of the length of the child array to the length of the parent array.

    Parameters:
        parent_array (list): The parent array.
        child_array (list): The child array.

    Returns:
        float: The weight of the child array.
    """
    return len(child_array) / len(parent_array)


def entropy(arr):
    """
    Calculates the entropy of a binary target class.

    Entropy is a measure of the uncertainty or impurity in a dataset. It is used in decision trees to determine the best split at each node.

    Parameters:
        arr (list of int): A list of binary values (0s and 1s) representing the target class.

    Returns:
        float: The entropy value, ranging from 0 (completely certain) to 1 (maximum uncertainty).

    Examples:
        >>> entropy([0, 0, 1, 1])
        1.0
        >>> entropy([0, 0, 0, 0])
        0.0
    """
    p = sum(arr) / len(arr)
    if p == 0 or p == 1:
        return 0  # Directly return 0 as there is complete certainty
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def information_gain(parent_entropy, weighted_entropy):
    """
    Calculate the information gain.

    Information gain is the reduction in entropy from the parent node to the child nodes.
    It is used to determine the best feature and threshold for splitting the data in decision trees.

    Parameters:
        parent_entropy (float): The entropy of the parent node.
        weighted_entropy (float): The weighted entropy of the child nodes.

    Returns:
        float: The information gain.
    """
    return parent_entropy - weighted_entropy


def weighted_entropy(child_entropies: list, child_weights: list):
    """
    Calculate the weighted entropy of child nodes.

    This function takes the entropies and weights of child nodes and calculates the
    weighted sum of the entropies. This is used to determine the overall entropy
    after a split in decision tree algorithms.

    Parameters:
        child_entropies (list of float): A list containing the entropy values of the child nodes.
        child_weights (list of float): A list containing the weights of the child nodes.

    Returns:
        float: The weighted entropy of the child nodes.
    """
    child_entropy_1 = child_entropies[0]
    child_entropy_2 = child_entropies[1]
    child_weight_1 = child_weights[0]
    child_weight_2 = child_weights[1]

    return child_weight_1 * child_entropy_1 + child_weight_2 * child_entropy_2


def find_best_split(feature_arrays, target_arr):
    """
    Identify the best feature and threshold for splitting the data.

    This function iterates over all features and their possible thresholds to find the split
    that results in the lowest weighted entropy. The best split is determined by comparing
    the weighted entropy of the child nodes for each possible split.

    Parameters:
        feature_arrays (numpy.ndarray): A 2D array where each column represents a feature.
        target_arr (numpy.ndarray): A 1D array representing the target variable.

    Returns:
        tuple: A tuple containing the best weighted entropy, the best threshold, and the index of the best feature.
    """
    weighted_entropy_best = 10000000
    feature_best = 0
    feature_threshold_best = 0
    num_features = feature_arrays.shape[1]

    for feature_ind in range(num_features):
        feature_arr = feature_arrays[:, feature_ind]
        feature_thresholds = thresholds(feature_arr)

        for threshold_ind in range(len(feature_thresholds)):

            child_arr_1 = target_arr[feature_arr > feature_thresholds[threshold_ind]]
            child_arr_2 = target_arr[feature_arr < feature_thresholds[threshold_ind]]

            child_entropy_1 = entropy(child_arr_1)
            child_entropy_2 = entropy(child_arr_2)

            child_weight_1 = weights(target_arr, child_arr_1)
            child_weight_2 = 1 - child_weight_1

            weighted_entropy_current = weighted_entropy(
                [child_entropy_1, child_entropy_2],
                [child_weight_1, child_weight_2],
            )

            if weighted_entropy_current < weighted_entropy_best:
                weighted_entropy_best = weighted_entropy_current
                feature_threshold_best = feature_thresholds[threshold_ind]
                feature_best = feature_ind

    return weighted_entropy_best, feature_threshold_best, feature_best


# def create_leaf(leaf_node, feature=None, threshold=None, class_value=None):
#     """
#     Creates a dictionary representing a node or leaf in a decision tree.

#     Parameters:
#         leaf_node (int): Indicator of whether the node is a decision node (0) or a leaf node (1).
#         feature (str, optional): The feature used for splitting at the decision node. Default is None.
#         threshold (float, optional): The threshold value for the feature at the decision node. Default is None.
#         class_value (any, optional): The class value for the leaf node. Default is None.

#     Returns:
#         dict: A dictionary representing the node or leaf with relevant information.
#     """
#     if leaf_node == 0:
#         leaf_dictionary = {
#             "leaf_node": "node",
#             "feature": feature,
#             "threshold": threshold,
#             "left": None,
#             "right": None,
#         }
#     elif leaf_node == 1:
#         leaf_dictionary = {"leaf_node": "leaf", "class": class_value}

#     return leaf_dictionary


def build_tree(feature_arrays, target_arr, max_depth=10):
    """
    # add function to travel down branches for loop and return variables to split the data
    # split feature and target data, find the best split, create leaf
    """

    weighted_entropy_best, feature_threshold_best, feature_best = find_best_split(
        feature_arrays, target_arr
    )

    if max_depth == 0 or weighted_entropy_best == 0:
        counts = np.bincount(target_arr)
        return {"leaf_node": 1, "class": np.argmax(counts)}

    max_depth -= 1

    # create function from here
    mask_left = feature_arrays[:, feature_best] > feature_threshold_best
    mask_right = feature_arrays[:, feature_best] < feature_threshold_best

    left_features = feature_arrays[mask_left]
    right_features = feature_arrays[mask_right]

    left_target = target_arr[mask_left]
    right_target = target_arr[mask_right]

    tree = {
        "leaf_node": 0,
        "feature": feature_best,
        "threshold": feature_threshold_best,
        "left": build_tree(left_features, left_target, max_depth),
        "right": build_tree(right_features, right_target, max_depth),
    }

    return tree
