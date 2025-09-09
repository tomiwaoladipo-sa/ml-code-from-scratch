import numpy as np

x = [
    np.array(
        [
            [1.0, 2.0, 3.0],  # Sample 1
            [4.0, 5.0, 6.0],  # Sample 2
        ]
    ),  # Shape: (2, 3)
    np.array(
        [
            [1.0, 3.0, 3.0],  # Sample 1
            [4.0, 5.0, 6.0],  # Sample 2
        ]
    ),  # Shape: (2, 3)
]


def rev(list):
    y = list.copy()
    return y.reverse()

y = rev(x)
print(y)
