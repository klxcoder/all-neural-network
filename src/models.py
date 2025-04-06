class Sequential:
    def __init__(self):
        self.x = None
        self.y = None
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

models = type('models', (object,), {
    'Sequential': Sequential,
})