class Dense:
    def __init__(self, n: int, activation: str):
        """
        :param n: number of neurons
        :param activation: 'relu' | 'softmax'
        """
        print('init Dense model:', n, activation)

layers = type('models', (object,), {
    'Dense': Dense,
})