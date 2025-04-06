from simple_load_linear_regression import simple_load_linear_regression
from models import models
from layers import layers

x, y = simple_load_linear_regression()
x = x.reshape((-1, 1))

model = models.Sequential()
model.add(layers.Dense(1, 'relu'))
model.add(layers.Dense(1, 'softmax'))
model.compile()

model.fit(x, y)