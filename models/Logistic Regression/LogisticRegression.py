import math
import random

import numpy as np


class LogisticRegression:

    def __init__(self, threshold=0.00001, learning_rate=0.01):
        self.coefficients = None
        self.threshold = threshold
        self.learning_rate = learning_rate

    def fit(self, X, y):

        # initialise coefficients in numpy array
        self.coefficients = np.zeros(X.shape[1] + 1).reshape(1, X.shape[1] + 1)
        cost_current = 100000
        cost_previous = 0

        # column of ones to beginning of dataset
        # this is to enable clean matrix manipulation
        X = np.c_[np.ones(X.shape[0]), X]

        # ensure y numy array is in the right structure
        y_reshaped = y.reshape(1, y.shape[0])

        iter = 0
        # while abs(cost_current - cost_previous) > self.threshold:
        while iter < 1000:
            iter += 1

            cost_previous = cost_current
            # matrix multiplication to return predictions -> w*x, then reshaped into correct orientation
            predictions = self._sigmoid(X)

            # obtain length of data for normalisation
            n = y_reshaped.shape[1]

            # cost function which is the mse
            # 1/n * (y - y_hat)^2
            # this is translated into matrix multiplication
            cost_current = self._cost_function(y_reshaped, predictions)

            # gradient calculated which is based on the derivative of the MSE w.r.t to weights
            gradient = self._gradient(y_reshaped, predictions, X)

            # coefficients are adjusted based on the gradient * learning rate -> gradient descent
            self.coefficients -= gradient * self.learning_rate
        print("iter: ", iter)

    def _gradient(self, y, predictions, X):
        n = y.shape[1]
        return -1 / n * ((y - predictions) @ X)

    def _cost_function(self, y, predictions):
        n = y.shape[1]
        self._adjust_predictions(predictions)
        y_log_multi_output = np.log2(
            self._adjust_predictions(predictions)
        ) @ y.T + np.log2(self._adjust_predictions(1 - predictions)) @ (1 - y.T)
        return -1 / n * np.sum(y_log_multi_output)

    def _adjust_predictions(self, predictions):
        predictions_adjusted = predictions + 0.0000000000001
        predictions = np.where(predictions == 0, predictions_adjusted, predictions)
        return predictions

    def _sigmoid(self, X):
        return 1 / (1 + np.exp(-np.dot(self.coefficients, X.T)))

    def predict(self, X_test):
        X_test = np.c_[np.ones(X_test.shape[0]), X_test]
        return self._sigmoid(X_test)
