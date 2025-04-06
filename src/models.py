import numpy as np

np.random.seed(1)

def loss(y, y_pred):
    mse = (1 / len(y)) * sum((y - y_pred) ** 2)
    return mse

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
            # weight = np.random.uniform(-1, 1, (self.layers[layer_index].n, self.layers[layer_index+1].n))
            weight = np.zeros((self.layers[layer_index].n, self.layers[layer_index + 1].n))
            self.weights.append(weight)
    def forward(self, x):
        self.layers[0].neurons = np.array(x)
        for layer_index in range(len(self.layers) - 1):
            self.layers[layer_index + 1].neurons = np.dot(self.layers[layer_index].neurons, self.weights[layer_index]) + self.layers[layer_index + 1].biases
    def fit(self, x, y, learning_rate = 0.001, iterations = 1500):
        """
        :param x: input (features)
        :param y: output (labels)
        :param learning_rate: learning_rate
        :param iterations: iterations
        :return:
        """
        self.x = x
        self.y = y

        loss_history = []

        for i in range(iterations):
            self.forward(x)
            for layer_index in range(len(self.layers) - 2, -1, -1):
                y_pred = self.layers[layer_index+1].neurons.flatten()
                l = loss(y, y_pred)
                dloss_dw = (2 / x.shape[0]) * np.dot(x.T, (y_pred - y))
                dloss_db = (2 / x.shape[0]) * np.sum(y_pred - y)
                self.weights[layer_index] = self.weights[layer_index] - learning_rate * dloss_dw
                self.layers[layer_index+1].biases = self.layers[layer_index+1].biases - learning_rate * dloss_db
                if layer_index == len(self.layers) - 2:
                    loss_history.append(l)
        return  loss_history

    def add(self, layer):
        if len(self.layers) == 0 and layer.activation != '':
            raise ValueError("Should not apply activation function to the input layer")
        self.layers.append(layer)

models = type('models', (object,), {
    'Sequential': Sequential,
})