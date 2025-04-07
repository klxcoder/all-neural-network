from models2 import Dense, Sequential
import numpy as np

"""
How to add input size into model?

"""

def test_models2():
    model = Sequential(2)
    model.add(Dense(4))
    model.add(Dense(3, 'relu'))
    model.add(Dense(2, 'softmax'))
    model.compile()
    # ----------------w0------------------
    assert model.weights[0].shape == (4, 3)
    w0 = np.array([
        [0, 3, 4],
        [0, 5, -5],
        [-4, 2, 1],
        [4, -3, -1],
    ])
    model.weights[0] = w0
    # ----------------w1------------------
    assert  model.weights[1].shape == (3, 2)
    w1 = np.array([
        [4, 4],
        [2, 1],
        [4, -4],
    ])
    model.weights[1] = w1
    # ----------------b0------------------
    assert model.biases[0].shape == (2, 3)
    b0 = np.array([
        [2, 4, -1],
        [-3, -1, 2],
    ])
    model.biases[0] = b0
    # ----------------b1------------------
    assert model.biases[1].shape == (2, 2)
    b1 = np.array([
        [-5, -4],
        [3, 3],
    ])
    model.biases[1] = b1
    # ----------------x0------------------
    assert model.layers[0].input.shape == (2, 4)
    # forward
    x0 = np.array([
        [3, 2, -2, 1],
        [0, 4, 4, -2],
    ])
    model.forward(x0)
    # test model.layers[0].input
    assert (model.layers[0].input == x0).all()
    # test model.layers[0].output
    assert (model.layers[0].output == x0).all()
    # test model.layers[1].input
    x1 = np.array([
        [14, 16, -2],
        [-27, 33, -12],
    ])
    assert (model.layers[1].input == x1).all()
    # test model.layers[1].output
    a1 = np.array([
        [14, 16, 0],
        [0, 33, 0],
    ])
    assert (model.layers[1].output == a1).all()
    # test model.layers[2].input
    x2 = np.array([
        [83, 68],
        [69, 36],
    ])
    assert (model.layers[2].input == x2).all()
    # test model.layers[2].output
    a2 = np.array([
        [9.99999694e-01, 3.05902227e-07],
        [1.00000000e+00, 4.65888615e-15],
    ])
    assert np.allclose(model.layers[2].output, a2)