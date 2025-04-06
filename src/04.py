from simple_load_linear_regression import simple_load_linear_regression
from models import models
from layers import layers
import matplotlib.pyplot as plt

x, y = simple_load_linear_regression()

plt.scatter(x.flatten(), y)

x = x.reshape((-1, 1))

model = models.Sequential()
model.add(layers.Dense(1, 'relu'))
model.add(layers.Dense(1, 'softmax'))
model.compile()

model.fit(x, y)

y_pred = model.layers[-1].neurons.flatten()
plt.plot(x.flatten(), y_pred)

plt.show()