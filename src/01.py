import pandas as pd
from sklearn.datasets import load_iris
from sklearn.utils import Bunch
import numpy as np
from numpy.typing import NDArray

if __name__ == "__main__":
    # Load the Iris dataset
    iris: Bunch = load_iris()
    X: NDArray[np.float64] = iris.data
    y: NDArray[np.int64] = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    # Access the features (sepal length, sepal width, petal length, petal width)
    print("Features shape:", X.shape)
    print("Feature names:", feature_names)
    print("\nFirst 5 rows of features:\n", X[:5])

    # Access the target variable (species: 0=setosa, 1=versicolor, 2=virginica)
    print("Target shape:", y.shape)
    print("Target names:", target_names)
    print("\nFirst 5 target values:\n", y[:5])

    # Access the description of the dataset
    print("\nDescription:", iris.DESCR)

    # You can also load it as a Pandas DataFrame directly
    iris_df_bunch: Bunch = load_iris(as_frame=True)
    iris_df: pd.DataFrame = iris_df_bunch.frame
    print("\nIris DataFrame keys:\n", iris_df_bunch.keys())
    print("\nIris DataFrame:\n", iris_df.head())