from simple_load_linear_regression import simple_load_linear_regression
import matplotlib.pyplot as plt

def main():
    x, y = simple_load_linear_regression()
    print(x)
    print(y)
    plt.scatter(x, y)
    plt.show()

if __name__ == "__main__":
    main()