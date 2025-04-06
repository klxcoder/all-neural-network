class Sequential:
    def __init__(self):
        print('init Sequential model')

models = type('models', (object,), {
    'Sequential': Sequential,
})