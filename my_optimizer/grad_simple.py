import numpy as np


def grad(x):
    grad = np.zeros((x.size, 1))
    for i in range(x.size):
        grad[i] = 4 * x[i]**3
    return grad