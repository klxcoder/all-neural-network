from simple_load_linear_regression import simple_load_linear_regression
from models import models
from layers import layers
import matplotlib.pyplot as plt

def main():
    x, y = simple_load_linear_regression()
    x = x.reshape((-1, 1))
    model = models.Sequential()
    model.add(layers.Dense(1, 'relu'))
    model.add(layers.Dense(1, 'softmax'))
    model.compile()

    _, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].scatter(x.flatten(), y, color='blue', label='Original data')
    axes[0].set_title('y = f(x)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    loss_history = model.fit(x, y)
    y_pred = model.layers[-1].neurons.flatten()
    axes[0].plot(x.flatten(), y_pred, color='red', label='Fitted line')
    axes[0].legend()

    axes[1].set_title('loss')
    axes[1].set_xlabel('t')
    axes[1].set_ylabel('loss')
    axes[1].plot(loss_history)

    plt.show()
if __name__ == "__main__":
    main()