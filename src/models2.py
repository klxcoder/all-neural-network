"""
Better version of models.py
Can forward and backpropagation (gradient descent)
Can apply activation function: ReLU or softmax
"""

import numpy as np

np.random.seed(1)

class Sequential:
    def __init__(self):
        # array of dict layers
        # include input layer (layer 0) and output layer (layer n+1)
        # each layer contains dict {n, input, activation, output}
        self.layers = []
        # array of weights
        self.weights = []
        # array of biases
        self.biases = []
        # The number of hidden layers
        self.n = 0
    def add(self, layer):
        if len(self.layers) == 0 and layer.activation != 'linear':
            raise ValueError("The input layer must use linear activation function")
        self.layers.append(layer)
    def compile(self):
        # Check layers length
        if len(self.layers) < 2:
            # There are at least 1 input layer and 1 output layer
            # So layers length must be at least 2
            raise ValueError("layers length must be at least 2")

        # Init weights and biases
        for i in range(self.n + 1):
            weight = np.zeros((self.layers[i].n, self.layers[i + 1].n))
            self.weights.append(weight)
            bias = np.zeros((self.layers[i].n, self.layers[i + 1].n))
            self.biases.append(bias)
    def forward(self, input):
        pass
    def fit(self, input, learning_rate = 0.001, iterations = 1500):
        pass