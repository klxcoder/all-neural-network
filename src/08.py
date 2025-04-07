import numpy as np

w0 = np.array([
    [0, 3, 4],
    [0, 5, -5],
    [-4, 2, 1],
    [4, -3, -1],
])
w1 = np.array([
    [4, 4],
    [2, 1],
    [4, -4],
])
b0 = np.array([
    [2, 4, -1],
    [-3, -1, 2],
])
b1 = np.array([
    [-5, -4],
    [3, 3],
])
x0 = np.array([
    [3, 2, -2, 1],
    [0, 4, 4, -2],
])

a0 = x0
# print(f"a0 = {a0}")
"""
a0 = [[ 3  2 -2  1]
 [ 0  4  4 -2]]
"""

# print(a0.shape) # (2, 4)
# print(w0.shape) # (4, 3)
# print(b0.shape) # (2, 3)

x1 = np.dot(a0, w0) + b0
# print(f"x1 = {x1}")
"""
x1 = [[ 14  16  -2]
 [-27  33 -12]]
"""

a1 = np.maximum(0, x1) # not np.max
# print(f"a1 = {a1}")
"""
a1 = [[14 16  0]
 [ 0 33  0]]
"""

x2 = np.dot(a1, w1) + b1
print(f"x2 = {x2}")
"""
x2 = [[83 68]
 [69 36]]
"""

# Function to apply softmax along the features axis (axis 1)
def softmax(x, axis=1):
    # Subtract the max for numerical stability
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

a2 = softmax(x2)
print(f"a2 = {a2}")
"""
a2 = [[9.99999694e-01 3.05902227e-07]
 [1.00000000e+00 4.65888615e-15]]
"""

sum_row_0 = np.sum(a2[0])
# print(f"sum_row_0 = {sum_row_0}")
# sum_row_0 = 0.9999999999999999

sum_row_1 = np.sum(a2[1])
# print(f"sum_row_1 = {sum_row_1}")
# sum_row_1 = 1.0