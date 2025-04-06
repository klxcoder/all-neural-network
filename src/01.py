# import pandas as pd
from sklearn.datasets import load_iris
from sklearn.utils import Bunch
# import numpy as np
# from numpy.typing import NDArray
from dataclasses import dataclass

@dataclass
class Data:
    data: list[list[float]]

# from pandas.api.types import CategoricalDtype

# Load the Iris dataset
iris: tuple[Bunch, tuple[Data, ...]] = load_iris(return_X_y=False)

print(iris)

# X: NDArray[np.float64] = iris.data

# print(X)

# y: NDArray[np.int64] = iris.target
# feature_names: List[str] = iris.feature_names
# target_names: List[str] = iris.target_names

# # The 'iris' object is a Bunch object, similar to a dictionary
# print(iris.keys())

# # Access the features (sepal length, sepal width, petal length, petal width)
# print("Features shape:", X.shape)
# print("Feature names:", feature_names)
# print("\nFirst 5 rows of features:\n", X[:5])

# # Access the target variable (species: 0=setosa, 1=versicolor, 2=virginica)
# print("Target shape:", y.shape)
# print("Target names:", target_names)
# print("\nFirst 5 target values:\n", y[:5])

# # Access the description of the dataset
# print("\nDescription:", iris.DESCR)

# # You can also load it as a Pandas DataFrame directly
# iris_df_tuple: Tuple[Bunch, Tuple[Any, ...]] = load_iris(as_frame=True)
# iris_df_bunch: Bunch = iris_df_tuple[0]
# iris_df: pd.DataFrame = iris_df_bunch.frame
# print("\nIris DataFrame keys:\n", iris_df_bunch.keys())
# print("\nIris DataFrame:\n", iris_df.head())


# # --- Pandas Example with Type Hints ---
# iris_loaded_tuple: Tuple[Bunch, Tuple[Any, ...]] = load_iris()
# iris_loaded: Bunch = iris_loaded_tuple[0]
# data: NDArray[np.float64] = iris_loaded.data
# target: NDArray[np.int64] = iris_loaded.target
# loaded_feature_names: List[str] = iris_loaded.feature_names
# loaded_target_names: List[str] = iris_loaded.target_names

# # Explicitly define the dtypes for the DataFrame
# dtypes: Dict[str, object] = {
#     loaded_feature_names[0]: np.float64,
#     loaded_feature_names[1]: np.float64,
#     loaded_feature_names[2]: np.float64,
#     loaded_feature_names[3]: np.float64,
#     'target': np.int64
# }

# df: pd.DataFrame = pd.DataFrame(data=np.c_[data, target], columns=loaded_feature_names + ['target'], dtype=dtypes)
# categorical_type = CategoricalDtype(categories=pd.Index(loaded_target_names))
# species_categorical: pd.Categorical = pd.Categorical.from_codes(target.astype(int).tolist(), dtype=categorical_type)
# df['species'] = pd.Series(species_categorical.astype(str))

# print("\nPandas DataFrame head:\n", df.head())
# print("\nDataFrame info:")
# print(df.info())
# print("\nSpecies distribution:")
# print(df['species'].value_counts())