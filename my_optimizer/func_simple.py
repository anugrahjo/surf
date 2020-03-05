def func(x):
    f = 0
    for i in range(x.size):
        f += x[i]**4
    return f