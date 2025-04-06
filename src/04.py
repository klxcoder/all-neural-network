from models import models
from layers import layers

model = models.Sequential()
dense = layers.Dense(10, 'relu')
model.add(dense)
model.fit([1, 2, 3], [2, 4, 6])