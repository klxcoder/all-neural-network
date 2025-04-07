import numpy as np

np.random.seed(1)

w0 = np.random.randint(-5, 5, size=(4, 3))
w1 = np.random.randint(-5, 5, size=(3, 2))

b0 = np.random.randint(-5, 5, size=(2, 3))
b1 = np.random.randint(-5, 5, size=(2, 2))

x0 = np.random.randint(-5, 5, size=(2, 4))

print("w0 = ", w0)
print("b0 = ", b0)

print("w1 = ", w1)
print("b1 = ", b1)

print("x0 = ", x0)