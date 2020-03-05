import numpy as np


def optimize(nx, x0, func, grad, tol):
    rho = 0.5
    eta = 1E-4
    grad1 = grad(x0)
    grad2 = grad1 * 1
    Bk = np.identity(nx)
    bfgs = 0
    norm_array = np.linalg.norm(grad1)
    while (np.linalg.norm(grad2) > tol):
        if bfgs > 1e+6:
            break
        bfgs += 1
        grad1 = grad2 * 1
        pk = np.linalg.solve(Bk, -grad1)
        alpha = 1
        while (func(x0 + alpha * pk) >
               (func(x0) + eta * alpha * np.matmul(grad1.T, pk))):
            alpha = rho * alpha

        sk = alpha * pk
        x0 += sk
        grad2 = grad(x0)
        yk = grad2 - grad1
        Bk = Bk + np.matmul(yk, yk.T) / np.matmul(yk.T, sk) - np.matmul(
            np.matmul(Bk, sk), np.matmul(sk.T, Bk)) / np.matmul(
                np.matmul(sk.T, Bk), sk)
        norm_array = np.append(norm_array, np.linalg.norm(grad2))
    iter_array = np.arange(bfgs + 1)

    return x0, iter_array, norm_array
