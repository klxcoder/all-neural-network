from models import models
from layers import layers
import numpy as np

model = models.Sequential()
model.add(layers.Dense(2, 'relu'))
model.add(layers.Dense(3, 'softmax'))
model.compile()
model.fit(np.random.uniform(-1, 1, (1, 2)), np.random.uniform(-1, 1, (1, 3)))