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
        """
        add layer to the neural network
        layer can be input layer, hidden layer, output layer
        """
        if len(self.layers) == 0 and layer.activation != 'linear':
            raise ValueError("The input layer must use linear activation function")
        # Add layer to the neural network
        self.layers.append(layer)
        # Update number of hidden layers
        self.n = max(0, len(self.layers) - 2)
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
        """
        forward input through neural network
        input should be a batch
        """
        self.layers[0].input = input
        self.layers[0].output = input
        for i in range(1, self.n + 2):
            cur_layer = self.layers[i]
            pre_layer = self.layers[i-1]
            cur_layer.input = np.dot(pre_layer.output, self.weights[i-1]) + self.biases[i-1]
            cur_layer.output = cur_layer.input
    def fit(self, input, learning_rate = 0.001, iterations = 1500):
        pass