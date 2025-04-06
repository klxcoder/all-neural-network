# simple linear regression

import numpy as np
import matplotlib.pyplot as plt

def get_data():
    x = np.arange(-10, 11)
    y = 3*x*x + 2*x + 5
    return x, y

def main():
    x, y = get_data()

    _, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].plot(x, y)
    axes[0].set_title('y = f(x)')

    plt.show()
if __name__ == "__main__":
    main()