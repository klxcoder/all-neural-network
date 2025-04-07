"""
Better version of models.py
Can forward and backpropagation (gradient descent)
Can apply activation function: ReLU or softmax
"""

from simple_load_linear_regression import simple_load_linear_regression

import numpy as np

np.random.seed(1)

class Dense:
    def __init__(self, n: int, activation: str = 'linear'):
        """
        :param n: number of neurons
        :param activation: 'linear' | 'relu' | 'softmax'
        """
        self.n = n
        self.activation = activation

class Sequential:
    def __init__(self, batch_size: int):
        # batch_size define Training Examples Number for Mini-Batch
        self.batch_size = batch_size
        # array of dict layers
        # include input layer (layer 0) and output layer (layer n+1)
        # each layer contains dict {n, input, activation, output}
        self.layers: list[Dense] = []
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
        # Init layer input and output
        layer.input = np.zeros((self.batch_size, layer.n))
        layer.output = np.zeros((self.batch_size, layer.n))
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
            bias = np.zeros((self.batch_size, self.layers[i + 1].n))
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
            # TODO: Use cur_layer.activation to update cur_layer.output
            if cur_layer.activation == 'relu':
                cur_layer.output = np.maximum(0, cur_layer.input)
            else:
                cur_layer.output = cur_layer.input # a = x for now
    def fit(self, input, output, learning_rate = 0.001, iterations = 1500):
        print('Will fit')
        print('input = ', input)
        print('output = ', output)
        print('learning_rate = ', learning_rate)
        print('iterations = ', iterations)
        for t in range(iterations):
            self.forward(input)

def main():
    x, y = simple_load_linear_regression()
    x = x.reshape((-1, 1))

    model = Sequential()
    model.add(Dense(1))
    model.add(Dense(1, 'softmax'))
    model.compile()
    model.fit(x, y, learning_rate=0.1, iterations=1)
    print(model.layers[-1].output)

if __name__ == "__main__":
    main()