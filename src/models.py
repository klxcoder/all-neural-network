class Sequential:
    def __init__(self):
        self.x = None
        self.y = None
        self.layers = []
        print('init Sequential model')
    def fit(self, x, y):
        """
        :param x: input (features)
        :param y: output (labels)
        :return:
        """
        self.x = x
        self.y = y
        print('Will fit:', x, y)
        print('layers')
        for layer in self.layers:
            print(layer)
    def add(self, layer):
        self.layers.append(layer)

models = type('models', (object,), {
    'Sequential': Sequential,
})