import numpy as np

class Dense:
    def __init__(self, n: int, activation: str):
        """
        :param n: number of neurons
        :param activation: 'relu' | 'softmax'
        """
        self.n = n
        self.neurons = np.zeros(n)
        self.biases = np.zeros(n)
    def __repr__(self):
        return f"layer: {len(self.neurons)} neurons and biases\n{self.neurons}\n{self.biases}"

layers = type('models', (object,), {
    'Dense': Dense,
})