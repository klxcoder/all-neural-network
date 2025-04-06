from models import models
from layers import layers

model = models.Sequential()
model.add(layers.Dense(2, 'relu'))
model.add(layers.Dense(1, 'softmax'))
model.compile()
# model.fit([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 2])
model.fit([[1, 1]], [2])