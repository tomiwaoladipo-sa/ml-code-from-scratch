import numpy as np
import pandas as pd
from LogisticRegression import LogisticRegression
from sklearn.datasets import load_breast_cancer

# Load the breast cancer dataset
breast_cancer = load_breast_cancer(as_frame=True)
df_breast_cancer = breast_cancer.data

# Convert the features dataframe to a numpy array
arr_features = np.array(df_breast_cancer.copy())

# Convert the target series to a numpy array
arr_target = np.array(breast_cancer.target.copy())


class LogisticRegression_test:

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
        pritn("iter: ", iter)

    def _gradient(self, y, predictions, X):
        n = y.shape[1]
        return -1 / n * np.sum((y - predictions) @ X)

    def _cost_function(self, y, predictions):
        n = y.shape[1]
        # print(n)
        y_log_multi_output = np.log2(predictions) @ y.T + np.log2(1 - predictions) @ (
            1 - y.T
        )
        return -1 / n * np.sum(y_log_multi_output)

    def _sigmoid(self, X):
        return 1 / (1 + np.exp(-np.dot(self.coefficients, X.T)))

    def predict(self, X_test):
        X_test = np.c_[np.ones(X_test.shape[0]), X_test]
        return self._sigmoid(X_test)


lr_test = LogisticRegression_test(threshold=0.000001, learning_rate=0.1)
lr_test.fit(arr_features, arr_target)
