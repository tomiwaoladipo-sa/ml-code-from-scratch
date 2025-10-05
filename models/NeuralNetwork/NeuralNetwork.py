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


def derivative_choice(string, obs=None, pred=None, X=None, w=None, z=None, y=None, layer=None):
    """
    This function takes a string and returns the corresponding derivative function.
    """
    if string == "dssr_dp":
        return -2 * (obs - pred)
    elif string == "dp_dw":
        return y
    elif string == "dp_dy":
        if layer == 0:
            w_output = w
        else:
            w_output = w[:-1, :] # remove bias weight
        return w_output
    elif string == "dy_dz":
        z_derivative = relu_derivative(z)
        return z_derivative
    elif string == "dz_dy":
        return w[:-1, :] # remove bias weight
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
                        derivative_choice("dp_dw", y=add_bias(y_lst_reverse[i])),
                    ]

                else:
                    derivative_dict[j] = ["dssr_dp", "dp_dy"]
                    derivative_dict_value[j] = [
                        derivative_choice("dssr_dp", obs=obs, pred=pred),
                        derivative_choice("dp_dy", obs=obs, w=weights_lst_reverse[i], layer=j),
                    ]
        if i > 0:
            for j in derivative_dict.keys():
                if j > 0 and j == i:
                    derivative_dict[j].append("dy_dz")
                    derivative_dict_value[j].append(
                        derivative_choice("dy_dz", obs=obs, z=z_lst_reverse[i - 1])
                    )
                    
                    if i == number_of_weight_layers - 1:
                        derivative_dict[j].append("dz_dw")
                        derivative_dict_value[j].append(derivative_choice("dz_dw", X=X))
                    else:
                        derivative_dict[j].append("dz_dw")
                        derivative_dict_value[j].append(
                            derivative_choice("dz_dw", X=add_bias(y_lst_reverse[i - 2]))
                        )

                elif j > 0 and j > i:

                    derivative_dict[j].append("dy_dz")
                    derivative_dict_value[j].append(
                        derivative_choice("dy_dz", obs=obs, z=z_lst_reverse[i - 1])
                    )
                    derivative_dict[j].append("dz_dy")
                    derivative_dict_value[j].append(
                        derivative_choice("dz_dy", obs=obs, w=weights_lst_reverse[i])
                    )
    return derivative_dict, derivative_dict_value

def add_bias(X):
    return np.c_[X, np.ones(X.shape[0])]


def generate_random_weights(X, size):
    # Generate random weights for the layer
    limit = np.sqrt(6 / (X.shape[1] + size))
    W = np.random.uniform(-limit, limit, (X.shape[1], size))
    # W = np.random.randn(X.shape[1], size)
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
            # print(current_input.shape, " x ", W.shape, " = ", y.shape)
            # 3. add bias
            # For hidden layers, add bias
            y_with_bias = np.c_[y, np.ones(y.shape[0])]

            # 4. add to the input list
            input_lst.append(y_with_bias)
        else:
            # For the output layer, no bias is added
            pred = Z

    return pred, input_lst, weights_lst, z_lst, y_lst, X_with_bias


def gradient(derivative_chain_values, derivative_chain, obs):
    derivative_product = None
    for val, deriv in zip(derivative_chain_values, derivative_chain):
        if derivative_product is not None:
            if deriv == "dy_dz":
                derivative_product = derivative_product*val
            elif deriv in ["dz_dw", "dp_dw"]:
                derivative_product = np.dot(derivative_product.T, val)
            else:
                derivative_product = np.dot(derivative_product, val.T)
        else:
            derivative_product = val
    return derivative_product/obs.shape[0]

def new_weights(weights, gradient, learning_rate=0.01):
    if weights.shape != gradient.shape:
        output = weights - gradient.T*learning_rate
    else: 
        output = weights - learning_rate*gradient
    return output
    

def gradient_descent(weights_lst, derivative_chain_values_lst, derivative_chain_lst, obs, learning_rate=0.001):
    new_weights_lst = []
    weight_layers = len(weights_lst)
    for i in range(weight_layers):
        gradient_i = gradient(derivative_chain_values_lst[weight_layers-i-1], derivative_chain_lst[weight_layers-i-1], obs)
        new_w = new_weights(weights_lst[i], gradient_i, learning_rate)
        new_weights_lst.append(new_w)
    return new_weights_lst

if __name__ == "__main__":
    from sklearn.datasets import fetch_california_housing
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_squared_error
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
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
    # Load the california Housing Dataset
    california = fetch_california_housing()

    X = california.data  # Features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    obs = california.target.reshape(-1, 1)
    target_scaler = StandardScaler()
    obs = target_scaler.fit_transform(obs)

    # Define the neural network architecture
    hidden_layer_size = [16, 8]

    hidden_layer_size_with_last = hidden_layer_size + [1]  # Add output layer size

    # Initial forward pass
    pred, input_lst, weights_lst, z_lst, y_lst, X_with_bias = forward_pass(
        X, hidden_layer_size_with_last
    )
    # Store predictions at each epoch
    predictions_over_epochs = []
    r2_epochs = []

    for iter in range(100):
        derivative_chain, derivative_chain_values = build_derivative_chain(
            weights_lst, obs, pred, X_with_bias, z_lst, y_lst
        )  

        new_weights_lst = gradient_descent(weights_lst, derivative_chain_values, derivative_chain, obs, learning_rate=0.1)

        pred, input_lst, weights_lst, z_lst, y_lst, X_with_bias = forward_pass(X, hidden_layer_size_with_last, weights=new_weights_lst)
        predictions_over_epochs.append(pred.copy())
        r2_epochs.append(round(r2_score(obs, pred),2))
        # print(round(r2_score(obs, pred), 2), " ", round(mean_squared_error(obs, pred)**(1/2), 2))
  
    # Prepare the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the true observations (static)
    ax.scatter(X[:, 0], target_scaler.inverse_transform(obs), color='black', label='True Observations', alpha=0.8)

    # Initialize the scatter plot for predictions
    pred_scatter = ax.scatter([], [], color='blue', label='Predictions', alpha=0.6)

    # Set labels and title
    ax.set_xlabel(f'Feature {0}')
    ax.set_ylabel('Target Value')
    ax.set_title('Predictions vs Observations Over Epochs')
    ax.legend()

    # Function to update the plot for each epoch
    def update(epoch):
        preds = predictions_over_epochs[epoch]
        r2 = r2_epochs[epoch]
        pred_scatter.set_offsets(np.c_[X[:, 0], target_scaler.inverse_transform(preds)])
        ax.set_title(f'Predictions vs Observations (Epoch {epoch} - R²: {r2})')

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(predictions_over_epochs), interval=200)

    # Show the animation
    plt.show()
