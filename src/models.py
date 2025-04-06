import numpy as np

np.random.seed(1)

class Sequential:
    def __init__(self):
        self.x = None
        self.y = None
        self.layers = []
        self.weights = []
    def compile(self):
        if len(self.layers) < 2:
            raise ValueError("layers length must be at least 2")
        for layer_index in range(len(self.layers) - 1):
            weight = np.random.uniform(-1, 1, (self.layers[layer_index].n, self.layers[layer_index+1].n))
            self.weights.append(weight)
    def forward(self, x):
        self.layers[0].neurons = np.array(x)
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
        self.forward(x)
        print(y)
        print(self.layers[-1].neurons)
    def add(self, layer):
        self.layers.append(layer)

models = type('models', (object,), {
    'Sequential': Sequential,
})