import numpy as np

np.random.seed(1)

class Sequential:
    def __init__(self):
        self.x = None
        self.y = None
        self.layers = []
        self.weights = []
        print('init Sequential model')
    def print_layers(self):
        print('layers')
        for layer in self.layers:
            print(layer)
    def compile(self):
        if len(self.layers) < 2:
            raise ValueError("layers length must be at least 2")
        for layer_index in range(len(self.layers) - 1):
            weight = np.random.uniform(-1, 1, (self.layers[layer_index].n, self.layers[layer_index+1].n))
            self.weights.append(weight)
        print(self.weights)
    def forward(self, x0):
        self.layers[0].neurons = x0
        for layer_index in range(len(self.layers) - 1):
            self.layers[layer_index + 1].neurons = np.dot(self.layers[layer_index].neurons, self.weights[layer_index])
    def fit(self, x, y):
        """
        :param x: input (features)
        :param y: output (labels)
        :return:
        """
        self.x = x
        self.y = y
        self.print_layers()
        self.forward(x[0])
        self.print_layers()
    def add(self, layer):
        self.layers.append(layer)

models = type('models', (object,), {
    'Sequential': Sequential,
})