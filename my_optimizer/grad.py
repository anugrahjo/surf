import numpy as np


def grad(x):
    grad = np.array([x[1] - x[0], x[0] - 1])
    return grad