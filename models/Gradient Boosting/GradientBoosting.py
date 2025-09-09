import math
import random

import numpy as np


class RandomForest:
    def __init__(self):
        self.tree = None
        self.estimators = []

    def fit(self, X, y, max_depth, num_features, max_estimators=5):
        data = np.append(X, y.reshape(-1, 1), axis=1)
        for i in range(max_estimators):

            # create a array of indices of the same size as the data, with replacement
            indices = np.random.choice(data.shape[0], size=data.shape[0], replace=True)
            # create a bootstrap sample using the indices
            bootstrap_sample = data[indices]
            X_sample = bootstrap_sample[:, :-1]
            y_sample = bootstrap_sample[:, -1].astype(int)
            self.estimators.append(
                self._build_tree(
                    X_sample, y_sample, max_depth, num_features=num_features
                )
            )
        return self.estimators

    def predict(self, X):
        predictions = []
        for estimator in self.estimators:
            predictions.append(self._predict(X, estimator))

        mean_predictions = np.mean(predictions, axis=0)
        bool_array = mean_predictions > 0.5
        tie_array = mean_predictions == 0.5

        final_prediction = np.zeros(len(predictions[0]))
        final_prediction[bool_array] = 1
        final_prediction[tie_array] = np.random.choice([0, 1], size=tie_array.sum())

        return final_prediction

    def _thresholds(self, arr):
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

    def _weights(self, parent_array, child_array):
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

    def _entropy(self, arr):
        """
        Calculates the entropy of a binary target class.

        Entropy is a measure of the uncertainty or impurity in a dataset. It is used in decision trees to determine the best split at each node.

        Parameters:
            arr (list of int): A list of binary values (0s and 1s) representing the target class.

        Returns:
            float: The entropy value, ranging from 0 (completely certain) to 1 (maximum uncertainty).

        Examples:
            >>> _entropy([0, 0, 1, 1])
            1.0
            >>> _entropy([0, 0, 0, 0])
            0.0
        """
        p = sum(arr) / len(arr)
        if p == 0 or p == 1:
            return 0  # Directly return 0 as there is complete certainty
        return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

    def _information_gain(self, parent_entropy, weighted_entropy):
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

    def _weighted__entropy(self, child_entropies: list, child_weights: list):
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

    def _find_best_split(self, feature_arrays, target_arr, feature_ids):
        """
        Finds the best feature and threshold to split the data to minimize the weighted entropy.

        Parameters:
        feature_arrays (numpy.ndarray): 2D array where each column represents a feature and each row represents an observation.
        target_arr (numpy.ndarray): 1D array representing the target values corresponding to the observations.

        Returns:
        tuple: A tuple containing:
            - weighted_entropy_best (float): The best (minimum) weighted entropy found.
            - feature_threshold_best (float): The threshold value of the best feature to split on.
            - feature_best (int): The index of the best feature to split on.
        """
        weighted_entropy_best = 10000000
        feature_best = 0
        feature_threshold_best = 0
        num_features = feature_arrays.shape[1]
        # feature_ids = [1]
        for feature_ind in range(num_features):
            if feature_ids is not None and feature_ind not in feature_ids:
                continue
            feature_arr = feature_arrays[:, feature_ind]
            feature_thresholds = self._thresholds(feature_arr)

            for threshold_ind in range(len(feature_thresholds)):

                child_arr_1 = target_arr[
                    feature_arr > feature_thresholds[threshold_ind]
                ]
                child_arr_2 = target_arr[
                    feature_arr < feature_thresholds[threshold_ind]
                ]

                child_entropy_1 = self._entropy(child_arr_1)
                child_entropy_2 = self._entropy(child_arr_2)

                child_weight_1 = self._weights(target_arr, child_arr_1)
                child_weight_2 = 1 - child_weight_1

                weighted_entropy_current = self._weighted__entropy(
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

    def _feature_random(self, num_features, arr_features):
        """
        Randomly select a subset of features to consider for splitting.

        Parameters:
            num_features (int): The total number of features in the dataset.

        Returns:
            list: A list of indices representing the randomly selected features.
        """

        return random.sample(range(1, arr_features.shape[1] + 1), num_features)

    def _build_tree(
        self,
        feature_arrays,
        target_arr,
        max_depth,
        parent_entropy=None,
        num_features=None,
    ):
        """
        Recursively build the decision tree.

        This function splits the data based on the best feature and threshold, and then
        recursively builds the left and right subtrees until the maximum depth is reached
        or no further splits can be made.

        Parameters:
            feature_arrays (numpy.ndarray): A 2D array where each column represents a feature.
            target_arr (numpy.ndarray): A 1D array representing the target variable.
            max_depth (int): The maximum depth of the tree.

        Returns:
            dict: A dictionary representing the decision tree.
        """
        feature_ids = (
            None
            if num_features is None
            else self._feature_random(num_features, feature_arrays)
        )

        weighted_entropy_best, feature_threshold_best, feature_best = (
            self._find_best_split(feature_arrays, target_arr, feature_ids)
        )
        if parent_entropy is not None:
            information_gain_best = self._information_gain(
                parent_entropy, weighted_entropy_best
            )
        else:
            information_gain_best = 10000

        if max_depth == 0 or len(target_arr) < 2 or information_gain_best <= 0.01:
            counts = np.bincount(target_arr)
            return {
                "leaf_node": 1,
                "entopy": self._entropy(target_arr),
                "parent_entropy": parent_entropy,
                "weighted_entropy": weighted_entropy_best,
                "information_gain": information_gain_best,
                "class_perc": sum(target_arr) / len(target_arr),
                "samples": len(target_arr),
                "class": np.argmax(counts),
            }

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
            "feature_ids": feature_ids,
            "feature": feature_best,
            "threshold": feature_threshold_best,
            "entropy": self._entropy(target_arr),
            "information_gain": information_gain_best,
            "left": self._build_tree(
                left_features,
                left_target,
                max_depth,
                self._entropy(left_target),
                num_features,
            ),
            "right": self._build_tree(
                right_features,
                right_target,
                max_depth,
                self._entropy(right_target),
                num_features,
            ),
            "samples": len(target_arr),
        }

        return tree

    # a recursive approach can be used instead of the while loop
    def _predict(self, arr_features, custom_tree):
        """
        Predict the class labels for the given features using a custom decision tree.

        Parameters:
        arr_features (numpy.ndarray): A 2D array where each row represents the features of a single sample.
        custom_tree (dict): A dictionary representing the decision tree. The tree should have the following structure:
            {
                "leaf_node": int,  # 1 if it's a leaf node, 0 otherwise
                "class": int,      # The class label if it's a leaf node
                "feature": int,    # The index of the feature to split on
                "threshold": float,# The threshold value to split the feature
                "left": dict,      # The left subtree (if not a leaf node)
                "right": dict      # The right subtree (if not a leaf node)
            }

        Returns:
        numpy.ndarray: An array of predicted class labels for each row in arr_features.
        """
        tree_copy = custom_tree.copy()
        leaf_node = tree_copy["leaf_node"]
        predictions = np.array([])

        for row in arr_features:
            tree_copy = custom_tree.copy()
            while leaf_node == 0:
                if tree_copy["leaf_node"] == 1:
                    predictions = np.append(predictions, tree_copy["class"])
                    break

                feature = tree_copy["feature"]
                threshold = tree_copy["threshold"]

                if row[feature] > threshold:
                    tree_copy = tree_copy["left"]
                else:
                    tree_copy = tree_copy["right"]
        return predictions
