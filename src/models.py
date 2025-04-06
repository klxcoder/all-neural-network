import numpy as np

np.random.seed(1)

def loss(y, y_pred):
    mse = (1 / len(y)) * sum((y - y_pred) ** 2)
    return  mse

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
            self.layers[layer_index + 1].neurons = np.dot(self.layers[layer_index].neurons, self.weights[layer_index])
    def fit(self, x, y):
        """
        :param x: input (features)
        :param y: output (labels)
        :return:
        """
        self.x = x
        self.y = y

        learning_rate = 0.001
        iterations = 15
        loss_history = []

        for i in range(iterations):
            self.forward(x)
            y_pred = self.layers[-1].neurons.flatten()
            l = loss(y, y_pred)
            print('loss = ', l)
            dloss_dw = (2 / x.shape[0]) * np.dot(x.T, (y_pred - y))
            print('dloss_dw = ', dloss_dw)
            dloss_db = (2 / x.shape[0]) * np.sum(y_pred - y)
            print('dloss_db = ', dloss_db)
            self.weights[0] = self.weights[0] - learning_rate * dloss_dw
            # db = db - learning_rate * dloss_db
            loss_history.append(l)
        return  loss_history

    def add(self, layer):
        self.layers.append(layer)

models = type('models', (object,), {
    'Sequential': Sequential,
})