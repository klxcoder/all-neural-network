import numpy as np

class Dense:
    def __init__(self, n: int, activation: str):
        """
        :param n: number of neurons
        :param activation: 'relu' | 'softmax'
        """
        self.neurons = np.zeros(n)
        self.biases = np.zeros(n)

layers = type('models', (object,), {
    'Dense': Dense,
})