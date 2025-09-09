import math
import random

import numpy as np


class LinearRegression:

    def __init__(self, threshold=0.000001, learning_rate=0.1):
        self.coefficients = None
        self.threshold = threshold
        self.learning_rate = learning_rate

    def fit(self, X, y):

        # initialise coefficients in numpy array
        self.coefficients = np.random.rand(X.shape[1] + 1).reshape(1, X.shape[1] + 1)
        cost_current = 100000
        cost_previous = 0

        # column of ones to beginning of dataset
        # this is to enable clean matrix manipulation
        X = np.c_[np.ones(X.shape[0]), X]

        # ensure y numy array is in the right structure
        y_reshaped = y.reshape(1, y.shape[0])

        iter = 0
        while abs(cost_current - cost_previous) > self.threshold or iter < 1000:
            iter += 1

            cost_previous = cost_current
            # matrix multiplication to return predictions -> w*x, then reshaped into correct orientation
            prediction = np.dot(self.coefficients, X.T).reshape(1, y_reshaped.shape[1])

            # obtain length of data for normalisation
            n = y_reshaped.shape[1]

            # cost function which is the mse
            # 1/n * (y - y_hat)^2
            # this is translated into matrix multiplication
            cost_current = (
                1 / n * np.dot((y_reshaped - prediction), (y_reshaped - prediction).T)
            )

            # gradient calculated which is based on the derivative of the MSE w.r.t to weights
            gradient = -2 / n * np.dot((y_reshaped - prediction), X)

            # coefficients are adjusted based on the gradient * learning rate -> gradient descent
            self.coefficients -= gradient * self.learning_rate

    def predict(self, X_test):
        X_test = np.c_[np.ones(X_test.shape[0]), X_test]
        return np.dot(self.coefficients, X_test.T)[0]
