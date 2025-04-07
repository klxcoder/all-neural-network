import numpy as np

np.random.seed(1)

w0 = np.random.randint(-5, 5, size=(4, 3))
b0 = np.random.randint(-5, 5, size=(4, 3))

w1 = np.random.randint(-5, 5, size=(3, 2))
b1 = np.random.randint(-5, 5, size=(3, 2))

x0 = np.random.randint(-5, 5, size=(2, 4))

print("w0 = ", w0)
print("b0 = ", b0)

print("w1 = ", w1)
print("b1 = ", b1)

print("x0 = ", x0)

"""
w0 =  [[ 0  3  4]
 [ 0 -5 -5]
 [-4  2  1]
 [ 4 -3 -1]]
b0 =  [[ 0 -3 -1]
 [-3 -1  2]
 [ 2  4 -4]
 [ 2 -5  1]]
w1 =  [[ 4  4]
 [ 2  1]
 [ 4 -4]]
b1 =  [[-5 -4]
 [ 3  3]
 [-2  4]]
x0 =  [[ 3  2 -2  1]
 [ 0 -4  4 -2]]
"""