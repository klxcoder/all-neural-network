# simple linear regression

import numpy as np
import matplotlib.pyplot as plt

def get_data():
    n = 10
    x = np.random.uniform(-1, 1, n) * 20
    y = 3 * x + 5 + np.random.uniform(-1, 1, n) * 5
    return x, y

def linear_regression(x, y, learning_rate=0.001, iterations=15):
    m = 0
    b = 0
    n = len(x)
    loss_history = []

    for i in range(iterations):
        y_pred = m * x + b
        m_gradient = (-2 / n) * sum(x * (y - y_pred))
        b_gradient = (-2 / n) * sum(y - y_pred)
        m = m - learning_rate * m_gradient
        b = b - learning_rate * b_gradient
        loss = (1 / n) * sum((y - y_pred) ** 2)
        loss_history.append(loss)

    return m, b, loss_history

def main():
    x, y = get_data()

    _, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].scatter(x, y, color='blue', label='Original data')
    axes[0].set_title('y = f(x)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    m, b, loss_history = linear_regression(x, y)
    axes[0].plot(x, m*x + b, color='red', label='Fitted line')
    axes[0].legend()

    axes[1].set_title('loss')
    axes[1].set_xlabel('t')
    axes[1].set_ylabel('loss')
    axes[1].plot(loss_history)

    plt.show()
if __name__ == "__main__":
    main()