import numpy as np
import sklearn.linear_model as sklm
from LinearRegression import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# definde mse function
def mse(y_pred, y_test):
    y_diff = y_pred - y_test
    y_sq = y_diff**2
    mse = np.sum(y_sq) / len(y_diff)
    return mse


california_housing = fetch_california_housing()
X, y = california_housing.data, california_housing.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
lr = LinearRegression(threshold=0.000001, learning_rate=0.1)
lr.fit(X_train_scaled, y_train)
lr_weights = lr.coefficients[0][1:]
lr_bias = lr.coefficients[0][0]

print("weights: ", lr_weights, "bias: ", lr_bias)

sk_lr = sklm.LinearRegression()
sk_lr.fit(X_train_scaled, y_train)

sk_weights = sk_lr.coef_
sk_bias = sk_lr.intercept_

print("is weights difference significant: ", abs(lr_weights - sk_weights) > 0.0001)
print("is bias difference significant: ", abs(lr_bias - sk_bias) > 0.0001)


# predictions
X_test_scaled = scaler.fit_transform(X_test)
y_pred_lr = lr.predict(X_test_scaled)
y_pred_sk = sk_lr.predict(X_test_scaled)


print("custom function mse: ", mse(y_pred_lr, y_test))
print("sklearn function mse: ", mse(y_pred_sk, y_test))
print("custom vs sklearn mse: ", mse(y_pred_lr, y_pred_sk))

print("y_custom_predictions: ", y_pred_lr)
print("y_skleanr_predictions: ", y_pred_sk)
print("y_test: ", y_test)
