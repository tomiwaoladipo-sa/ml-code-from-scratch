import itertools
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
    This function calculates the weights for the child arrays.
    """
    return len(child_array) / len(parent_array)


def entropy(arr):
    """
    This function calculates entropy for the target class data passed to the function.
    """
    p = sum(arr) / len(arr)
    if p == 0 or p == 1:
        return 0  # Directly return 0 for certainty
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def information_gain(parent_entropy, weighted_entropy):
    """
    This function calculates information gain.
    """
    return parent_entropy - weighted_entropy


def weighted_entropy(child_entropies: list, child_weights: list):
    """
    This function calculates the weighted entropy.
    """
    child_entropy_1 = child_entropies[0]
    child_entropy_2 = child_entropies[1]
    child_weight_1 = child_weights[0]
    child_weight_2 = child_weights[1]

    return child_weight_1 * child_entropy_1 + child_weight_2 * child_entropy_2


def find_best_split(feature_arrays, target_arr):
    """
    This code identifies the best feature split.
    """
    weighted_entropy_best = 10000000
    feature_best = 0
    feature_threshold_best = 0
    num_features = arr_features.shape[1]

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


def create_leaf(leaf_node, feature=None, threshold=None, class_value=None):
    """ """
    if leaf_node == 0:
        leaf_dictionary = {
            "leaf_node": "node",
            "feature": feature,
            "threshold": threshold,
            "left": None,
            "right": None,
        }
    elif leaf_node == 1:
        leaf_dictionary = {"leaf_node": "leaf", "class": class_value}

    return leaf_dictionary


def generate_unique_combinations_matrix(rows):
    """
    Generate a matrix with unique combinations of 0s and 1s for each column.

    Parameters:
    rows (int): Number of rows in the matrix.

    Returns:
    np.ndarray: A binary matrix of shape (rows, 2^rows) with unique column combinations.
    """
    # Generate all possible unique combinations for the given number of rows
    all_combinations = np.array(list(itertools.product([0, 1], repeat=rows)))
    return all_combinations.T  # Transpose so that columns represent unique combinations


def build_tree(feature_arrays, target_arr, max_depth=10):
    """
    # add function to travel down branches for loop and return variables to split the data
    # split feature and target data, find the best split, create leaf
    """

    # level 0 branch creation
    weighted_entropy_best, feature_threshold_best, feature_best = find_best_split(
        feature_arrays, target_arr
    )

    tree = create_leaf(
        leaf_node=0, feature=feature_best, threshold=feature_threshold_best
    )
    # create function from here
    mask_left = feature_arrays[:, feature_best] > feature_threshold_best
    mask_right = feature_arrays[:, feature_best] < feature_threshold_best
    left_features = feature_arrays[mask_left]
    right_features = feature_arrays[mask_right]
    left_target = target_arr[mask_left]
    right_target = target_arr[mask_right]

    left_weighted_entropy_best, left_feature_threshold_best, left_feature_best = (
        find_best_split(left_features, left_target)
    )

    right_weighted_entropy_best, right_feature_threshold_best, right_feature_best = (
        find_best_split(right_features, right_target)
    )

    tree["left"] = create_leaf(
        leaf_node=0, feature=left_feature_best, threshold=left_feature_threshold_best
    )

    tree["right"] = create_leaf(
        leaf_node=0, feature=right_feature_best, threshold=right_feature_threshold_best
    )

    # parent_entropy = entropy(target_arr)
    # child_entropy = weighted_entropy_best
    # depth = 0
    # directions = ["left", "right"]
    # while depth < max_depth:
    #     depth += 1
    #     num_branches = 2**depth
    #     branch_routes = generate_unique_combinations_matrix(rows=depth)
    #     leaf = tree

    #     for branch in range(num_branches):
    #         for direction in branch_routes[:, branch]:
    #             leaf = leaf[branch[directions[direction]]]
    #             if leaf == None: # if class == 1,0?
    #                 break
    #             feature = leaf["feature"]
    #             threshold = leaf["threshold"]

    # split features and target
    # find best split
    # create tree with left and right node
    # break out of while loop when no more branching can occur i.e. no more leaf nodes

    return tree
