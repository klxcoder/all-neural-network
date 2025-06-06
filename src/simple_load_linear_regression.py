import numpy as np

np.random.seed(1)

def simple_load_linear_regression():
    n = 10
    x = np.random.uniform(-1, 1, n) * 20
    y = 1 * x + 20 + np.random.uniform(-1, 1, n) * 5
    return x, y