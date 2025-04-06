# simple linear regression

import numpy as np
import matplotlib.pyplot as plt

def main():
    x = np.arange(10)
    _, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].plot(x, x * x)
    plt.show()
if __name__ == "__main__":
    main()