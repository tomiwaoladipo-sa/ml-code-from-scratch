import math
import random

import numpy as np

np.array([1, 1])


class GradientBoosting:
    def __init__(self):
        self.tree = None
        self.estimators = []
        self.output_value = 0.5
        self.learning_rate = 0.1
        self.probabilities_arr = []
        self.logit_arr = []
        self.output_value_arr = []

    def fit(
        self, X, y, max_depth=5, max_estimators=5, learning_rate=0.1, output_value=0.5
    ):
        self.output_value = output_value
        self.learning_rate = learning_rate
        logit_arr = self._log_odds(np.full(y.shape, output_value))
        # print(logit_arr)
        self.output_value_arr.append(logit_arr.copy())

        for i in range(max_estimators):
            # print(self.output_value_arr[0])
            probabilities_arr = self._sigmoid(logit_arr)
            residuals_arr = y - probabilities_arr
            tree = self._build_tree(
                residuals_arr, probabilities_arr, feature_arr=X, max_depth=max_depth
            )
            self.estimators.append(tree)
            # print(tree)
            # print(self._output_value_arr(X, tree))
            logit_arr += learning_rate * self._output_value_arr(X, tree)
            self.probabilities_arr.append(probabilities_arr.copy())
            self.logit_arr.append(logit_arr.copy())
            self.output_value_arr.append(self._output_value_arr(X, tree).copy())

        return self.estimators

    def predict(self, X):
        self.probabilities_arr = []
        logit_arr = self._log_odds(np.full(X.shape[0], self.output_value))
        self.probabilities_arr.append(self._sigmoid(logit_arr).copy())
        probabilities_arr = 0
        for estimator in self.estimators:
            logit_arr += self.learning_rate * self._output_value_arr(X, estimator)
            probabilities_arr = self._sigmoid(logit_arr)
            self.probabilities_arr.append(probabilities_arr.copy())

        return np.where(probabilities_arr >= 0.5, 1, 0)

    def _log_odds(self, probabilities_arr):
        return np.log(probabilities_arr / (1 - probabilities_arr))

    def _output_value_arr(self, X, tree):
        output_values = np.array([])
        for row in X:
            tree_copy = tree.copy()

            while True:
                if tree_copy["leaf_node"] == 1:
                    output_values = np.append(output_values, tree_copy["output value"])
                    break

                feature = tree_copy["feature"]
                threshold = tree_copy["threshold"]

                if row[feature] > threshold:
                    tree_copy = tree_copy["left"]
                else:
                    tree_copy = tree_copy["right"]

        return output_values

    def _sigmoid(self, x):
        # x = np.clip(x, -709, 709)
        return 1 / (1 + np.exp(-x))

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

    # def _weights(self, parent_array, child_array):
    #     """
    #     Calculate the weight of a child array relative to its parent array.

    #     The weight is defined as the ratio of the length of the child array to the length of the parent array.

    #     Parameters:
    #         parent_array (list): The parent array.
    #         child_array (list): The child array.

    #     Returns:
    #         float: The weight of the child array.
    #     """
    #     return len(child_array) / len(parent_array)

    def _similarity_score(self, residuals_arr, probabilities_arr):
        """ """
        return np.sum(residuals_arr) ** 2 / np.sum(
            probabilities_arr * (1 - probabilities_arr)
        )

    def _information_gain(self, parent_similarity_score, weighted_similarity_score):
        """
        Calculate the information gain.

        Information gain is the reduction in entropy from the parent node to the child nodes.
        It is used to determine the best feature and threshold for splitting the data in decision trees.

        Parameters:
            parent_similarity_score (float): The entropy of the parent node.
            weighted_similarity_score (float): The weighted entropy of the child nodes.

        Returns:
            float: The information gain.
        """
        return weighted_similarity_score - parent_similarity_score

    # def _weighted__similarity_score(self, child_entropies: list, child_weights: list):
    #     """
    #     Calculate the weighted entropy of child nodes.

    #     This function takes the entropies and weights of child nodes and calculates the
    #     weighted sum of the entropies. This is used to determine the overall entropy
    #     after a split in decision tree algorithms.

    #     Parameters:
    #         child_entropies (list of float): A list containing the entropy values of the child nodes.
    #         child_weights (list of float): A list containing the weights of the child nodes.

    #     Returns:
    #         float: The weighted entropy of the child nodes.
    #     """
    #     child_similarity_score_1 = child_entropies[0]
    #     child_similarity_score_2 = child_entropies[1]
    #     child_weight_1 = child_weights[0]
    #     child_weight_2 = child_weights[1]

    #     return child_weight_1 * child_similarity_score_1 + child_weight_2 * child_similarity_score_2

    def _find_best_split(self, residuals_arr, probabilities_arr, feature_arr):
        """
        Finds the best feature and threshold to split the data to minimize the weighted entropy.

        Parameters:
        feature_arrays (numpy.ndarray): 2D array where each column represents a feature and each row represents an observation.
        target_arr (numpy.ndarray): 1D array representing the target values corresponding to the observations.

        Returns:
        tuple: A tuple containing:
            - weighted_similarity_score_best (float): The best (minimum) weighted entropy found.
            - feature_threshold_best (float): The threshold value of the best feature to split on.
            - feature_best (int): The index of the best feature to split on.
        """
        weighted_similarity_score_best = 0
        feature_best = 0
        feature_threshold_best = 0
        num_features = feature_arr.shape[1]
        # feature_ids = [1]
        similarity_lst = []
        for feature_ind in range(num_features):

            feature = feature_arr[:, feature_ind]
            feature_thresholds = self._thresholds(feature)

            for threshold_ind in range(len(feature_thresholds)):

                residuals_child_arr_1 = residuals_arr[
                    feature > feature_thresholds[threshold_ind]
                ]
                residuals_child_arr_2 = residuals_arr[
                    feature < feature_thresholds[threshold_ind]
                ]

                probabilities_child_arr_1 = probabilities_arr[
                    feature > feature_thresholds[threshold_ind]
                ]
                probabilities_child_arr_2 = probabilities_arr[
                    feature < feature_thresholds[threshold_ind]
                ]

                child_similarity_score_1 = self._similarity_score(
                    residuals_child_arr_1, probabilities_child_arr_1
                )
                child_similarity_score_2 = self._similarity_score(
                    residuals_child_arr_2, probabilities_child_arr_2
                )

                # child_weight_1 = self._weights(target_arr, child_arr_1)
                # child_weight_2 = 1 - child_weight_1

                weighted_similarity_score_current = (
                    child_similarity_score_1 + child_similarity_score_2
                )
                similarity_lst.append(weighted_similarity_score_current)
                if weighted_similarity_score_current > weighted_similarity_score_best:
                    weighted_similarity_score_best = weighted_similarity_score_current
                    feature_threshold_best = feature_thresholds[threshold_ind]
                    feature_best = feature_ind
        # print(max(similarity_lst), min(similarity_lst), weighted_similarity_score_best)

        return weighted_similarity_score_best, feature_threshold_best, feature_best

    # def _feature_random(self, num_features, arr_features):
    #     """
    #     Randomly select a subset of features to consider for splitting.

    #     Parameters:
    #         num_features (int): The total number of features in the dataset.

    #     Returns:
    #         list: A list of indices representing the randomly selected features.
    #     """

    #     return random.sample(range(1, arr_features.shape[1] + 1), num_features)

    def _output_value(self, residuals_arr, probabilities_arr):
        """ """
        return np.sum(residuals_arr) / np.sum(
            probabilities_arr * (1 - probabilities_arr)
        )

    def _build_tree(self, residuals_arr, probabilities_arr, feature_arr, max_depth):
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

        weighted_similarity_score_best, feature_threshold_best, feature_best = (
            self._find_best_split(residuals_arr, probabilities_arr, feature_arr)
        )

        parent_similarity_score = self._similarity_score(
            residuals_arr, probabilities_arr
        )

        if parent_similarity_score is not None:
            information_gain_best = self._information_gain(
                parent_similarity_score, weighted_similarity_score_best
            )
        else:
            information_gain_best = 10000
        # print(information_gain_best, len(residuals_arr))
        if max_depth == 0 or len(residuals_arr) < 2 or information_gain_best <= 0.01:
            return {
                "leaf_node": 1,
                "entopy": self._similarity_score(residuals_arr, probabilities_arr),
                "parent_similarity_score": parent_similarity_score,
                "weighted_similarity_score": weighted_similarity_score_best,
                "information_gain": information_gain_best,
                "samples": len(residuals_arr),
                "output value": self._output_value(residuals_arr, probabilities_arr),
            }

        max_depth -= 1

        # create function from here
        mask_left = feature_arr[:, feature_best] > feature_threshold_best
        mask_right = feature_arr[:, feature_best] < feature_threshold_best

        left_features = feature_arr[mask_left]
        right_features = feature_arr[mask_right]

        left_residuals = residuals_arr[mask_left]
        right_residuals = residuals_arr[mask_right]

        left_probabilities = probabilities_arr[mask_left]
        right_probabilities = probabilities_arr[mask_right]

        tree = {
            "leaf_node": 0,
            "feature": feature_best,
            "threshold": feature_threshold_best,
            "entropy": parent_similarity_score,
            "information_gain": information_gain_best,
            "left": self._build_tree(
                left_residuals, left_probabilities, left_features, max_depth
            ),
            "right": self._build_tree(
                right_residuals, right_probabilities, right_features, max_depth
            ),
            "samples": len(residuals_arr),
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
        predictions = np.array([])
        for row in arr_features:
            tree = custom_tree.copy()

            while True:
                if tree["leaf_node"] == 1:
                    predictions = np.append(predictions, tree["output value"])
                    break

                feature = tree["feature"]
                threshold = tree["threshold"]

                if row[feature] > threshold:
                    tree = tree["left"]
                else:
                    tree = tree["right"]
        return predictions
