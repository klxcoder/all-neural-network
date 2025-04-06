from pyexpat import features

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.utils import Bunch
import numpy as np
from numpy.typing import NDArray

def simple_load_iris():
    """
    load_iris but simpler
    """
    # Load the Iris dataset
    iris: Bunch = load_iris()
    x: NDArray[np.float64] = iris.data
    y: NDArray[np.int64] = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    return {
        'x': x,
        'y': y,
        'feature_names': feature_names,
        'target_names': target_names,
    }

def main():
    x, y, feature_names, target_names = simple_load_iris().values()

    # Access the features (sepal length, sepal width, petal length, petal width)
    print("Features shape:", x.shape)
    print("Feature names:", feature_names)
    print("First 5 rows of features:\n", x[:5])

    print('_'*50)

    # Access the target variable (species: 0=setosa, 1=versicolor, 2=virginica)
    print("Target shape:", y.shape)
    print("Target names:", target_names)
    print("First 5 target values:\n", y[:5])

if __name__ == "__main__":
    main()