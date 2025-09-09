# I found addings probabilities using sigmoid is wrong

import matplotlib

# import matplotlib.pyplot as plt

# import numpy as np
# import pandas as pd
# from GradientBoosting_Rough import GradientBoosting

# # from RandomForest import RandomForest
# from sklearn.datasets import load_breast_cancer
# from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
# from sklearn.model_selection import train_test_split


# def count_depth(d):
#     if not isinstance(d, dict) or not d:
#         return 0
#     return 1 + max(count_depth(v) for v in d.values())


# # Load the breast cancer dataset
# breast_cancer = load_breast_cancer(as_frame=True)
# df_breast_cancer = breast_cancer.data

# df_breast_cancer = df_breast_cancer[["worst radius"]].copy()

# # Convert the features dataframe to a numpy array
# arr_features = np.array(df_breast_cancer.copy())

# # Convert the target series to a numpy array
# arr_target = np.array(breast_cancer.target.copy())

# # print(arr_target.mean())
# # Build the decision tree
# gb = GradientBoosting()
# custom_boosted_tree = gb.fit(
#     arr_features, arr_target, max_depth=5, max_estimators=20, learning_rate=0.2
# )

# sk_gb = GradientBoostingClassifier(
#     n_estimators=5, learning_rate=0.2, max_depth=5, random_state=0
# )
# sk_gb.fit(arr_features, arr_target)

# # Make predictions using the decision tree
# predictions = gb.predict(arr_features)
# # print(gb.probabilities_arr[0] == gb.probabilities_arr[2])
# # print(gb.logit_arr)
# # print(gb.probabilities_arr)
# # Print the number of misclassifications
# print(
#     "custom misclassifications %: ",
#     (len(arr_target) - sum(predictions == arr_target)) / len(arr_target) * 100,
#     "custom misclassifications abs: ",
#     len(arr_target) - sum(predictions == arr_target),
# )

# print(
#     "sk misclassifications %: ",
#     (len(arr_target) - sum(sk_gb.predict(arr_features) == arr_target))
#     / len(arr_target)
#     * 100,
#     "sk misclassifications abs: ",
#     len(arr_target) - sum(sk_gb.predict(arr_features) == arr_target),
# )


# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # Create bins for 'worst radius' to group data
# bins = np.linspace(
#     df_breast_cancer["worst radius"].min(), df_breast_cancer["worst radius"].max(), 20
# )
# df_breast_cancer["worst_radius_bin"] = pd.cut(df_breast_cancer["worst radius"], bins)

# # Calculate the percentage of malignant cases in each bin
# malignant_percentage = df_breast_cancer.groupby("worst_radius_bin").apply(
#     lambda df: (arr_target[df.index] == 1).mean() * 100
# )

# # Calculate the mean worst radius for each bin to plot on the x-axis
# bin_centers = df_breast_cancer.groupby("worst_radius_bin")["worst radius"].mean()

# # Plot setup
# fig, ax = plt.subplots(figsize=(10, 6))
# scatter_malignant = ax.scatter(
#     bin_centers,
#     malignant_percentage,
#     s=100,
#     color="blue",
#     edgecolors="black",
#     label="Actual Malignant Percentage",
# )
# scatter_predictions = ax.scatter(
#     arr_features,
#     gb.probabilities_arr[0] * 100,
#     s=100,
#     color="red",
#     edgecolors="black",
#     label="Model Predictions",
# )

# # Add plot details
# ax.set_title("Convergence of Gradient Boosting Predictions")
# ax.set_xlabel("Worst Radius")
# ax.set_ylabel("Malignant Percentage (%)")
# ax.grid(True)
# ax.legend()


# # Animation function
# def update(frame):
#     scatter_predictions.set_offsets(
#         np.c_[arr_features, gb.probabilities_arr[frame] * 100]
#     )
#     ax.set_title(f"Convergence of Gradient Boosting - Iteration {frame + 1}")
#     return (scatter_predictions,)


# # Create the animation
# ani = FuncAnimation(
#     fig,
#     update,
#     frames=len(gb.probabilities_arr),
#     interval=800,
#     blit=True,
#     repeat=True,  # Loop the animation
# )

# plt.tight_layout()
# plt.show()
