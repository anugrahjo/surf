import numpy as np
from func import func
from grad import grad
from ajoshy_optimize import optimize
import pandas

x0 = np.full((10, 1), 0.1)
x, iter_array, norm_array = optimize(10, x0, func, grad, 1E-7)

table = pandas.DataFrame({"Iter": iter_array, "Opt": norm_array})
print(table.to_string(index=False))
print(x)