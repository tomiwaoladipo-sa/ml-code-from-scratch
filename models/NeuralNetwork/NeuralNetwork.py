import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def dssr_dp(obs, pred):
    """
    obs: observed values
    pred: predicted values
    """
    return -2 * (obs - pred)


def dp_dw(x):
    return x


def dp_dy(w):
    return w


def dy_dz(z):
    return relu_derivative(z)


def dz_dy(w):
    return w


def dz_dw(x):
    return x


def derivative_choice(string, obs=None, pred=None, X=None, w=None, z=None, y=None):
    """
    This function takes a string and returns the corresponding derivative function.
    """
    if string == "dssr_dp":
        return -2 * (obs - pred)
    elif string == "dp_dw":
        return y
    elif string == "dp_dy":
        return w
    elif string == "dy_dz":
        return relu_derivative(z)
    elif string == "dz_dy":
        return w
    elif string == "dz_dw":
        return X
    else:
        raise ValueError(f"Unknown derivative: {string}")


def build_derivative_chain(weights_lst, obs, pred, X, z_lst, y_lst):
    weights_lst_reverse = weights_lst[::-1]
    y_lst_reverse = y_lst[::-1]
    z_lst_reverse = z_lst[::-1]

    derivative_dict = {}
    derivative_dict_value = {}
    number_of_weight_layers = len(weights_lst)
    for i in range(number_of_weight_layers):
        derivative_dict[i] = None

    for i in range(number_of_weight_layers):
        if i == 0:
            for j in derivative_dict.keys():
                if j == 0:
                    derivative_dict[j] = ["dssr_dp", "dp_dw"]
                    derivative_dict_value[j] = [
                        derivative_choice("dssr_dp", obs=obs, pred=pred),
                        derivative_choice("dp_dw", y=y_lst_reverse[i]),
                    ]

                else:
                    derivative_dict[j] = ["dssr_dp", "dp_dy"]
                    derivative_dict_value[j] = [
                        derivative_choice("dssr_dp", obs=obs, pred=pred),
                        derivative_choice("dp_dy", w=weights_lst_reverse[i]),
                    ]
        if i > 0:
            for j in derivative_dict.keys():
                if j > 0 and j == i:

                    derivative_dict[j].append("dy_dz")
                    derivative_dict_value[j].append(
                        derivative_choice("dy_dz", z=z_lst_reverse[i - 1])
                    )
                    derivative_dict[j].append("dz_dw")
                    if i == number_of_weight_layers - 1:
                        derivative_dict_value[j].append(derivative_choice("dz_dw", X=X))
                    else:
                        derivative_dict_value[j].append(
                            derivative_choice("dz_dw", X=y_lst_reverse[i - 2])
                        )

                elif j > 0 and j > i:

                    derivative_dict[j].append("dy_dz")
                    derivative_dict_value[j].append(
                        derivative_choice("dy_dz", z=z_lst_reverse[i - 1])
                    )
                    derivative_dict[j].append("dz_dy")
                    derivative_dict_value[j].append(
                        derivative_choice("dz_dy", w=weights_lst_reverse[i])
                    )
    return derivative_dict, derivative_dict_value


hidden_layer_size = [4, 3]

# Sample input data: 2 samples, 3 features each
X = np.array(
    [
        [1.0, 2.0, 3.0],  # Sample 1
        [4.0, 5.0, 6.0],  # Sample 2
    ]
)  # Shape: (2, 3)

obs = np.array(
    [
        [0],
        [1.0],  # Sample 1 observed value
    ]
)  # Observed values (2 samples)

# X_with_bias = np.c_[X, np.ones(X.shape[0])]  # Add bias term (1) to the input data

# n_rows, n_cols = X.shape

hidden_layer_size_with_last = hidden_layer_size + [1]  # Add output layer size


def add_bias(X):
    return np.c_[X, np.ones(X.shape[0])]


def generate_random_weights(X, size):
    # Generate random weights for the layer
    W = np.random.randn(X.shape[1], size)
    return W


def forward_pass(X, hidden_layer_size_with_last, weights=None):
    X_with_bias = add_bias(X)
    input_lst = [X_with_bias]
    weights_lst = []
    z_lst = []
    y_lst = []
    pred = None
    for i, size in enumerate(hidden_layer_size_with_last):
        current_input = input_lst[i]
        # Create weights for the current layer
        if weights is None:
            W = generate_random_weights(current_input, size)
        else:
            W = weights[i]
        weights_lst.append(W)
        # 1. multiply the input data with the weights
        Z = np.dot(current_input, W)
        if i < len(hidden_layer_size_with_last) - 1:
            z_lst.append(Z)
            # 2. apply the activation function (ReLU)
            y = relu(Z)
            y_lst.append(y)
            print(current_input.shape, " x ", W.shape, " = ", y.shape)
            # 3. add bias
            # For hidden layers, add bias
            y_with_bias = np.c_[y, np.ones(y.shape[0])]

            # 4. add to the input list
            input_lst.append(y_with_bias)
        else:
            # For the output layer, no bias is added
            pred = Z

    return pred, input_lst, weights_lst, z_lst, y_lst, X_with_bias


pred, input_lst, weights_lst, z_lst, y_lst, X_with_bias = forward_pass(
    X, hidden_layer_size_with_last
)


derivative_chain, derivative_chain_values = build_derivative_chain(
    weights_lst, obs, pred, X_with_bias, z_lst, y_lst
)  # dz/dw shuldn't be X only it can be y from before
print(derivative_chain, "derivative chain")
print(derivative_chain_values[1], "derivative chain values")
print("y:", y_lst)
# print(X_with_bias)
