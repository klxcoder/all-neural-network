from models import models
from layers import layers

model = models.Sequential()
model.add(layers.Dense(1, 'relu'))
model.add(layers.Dense(1, 'softmax'))
model.fit([1, 2, 3], [2, 4, 6])