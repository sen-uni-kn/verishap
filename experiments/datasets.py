# Copyright 2025 David Boetius
from pathlib import Path

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo


def categorical_to_int(x: pd.DataFrame) -> pd.DataFrame:
    for col in x.select_dtypes(include=["object"]).columns:
        x.loc[:, col] = x[col].astype("category").cat.codes
    return x


def save_dataset(dataset_name: str):
    def decorator(fn):
        def wrapper(*args, root=".datasets", **kwargs):
            file = Path(root) / f"{dataset_name}.npz"
            if file.exists():
                x = np.load(file)["x"]
                y = np.load(file)["y"]
                return x, y
            else:
                x, y = fn(*args, **kwargs)
                np.savez(Path(root) / f"{dataset_name}.npz", x=x, y=y)
                return x, y
        return wrapper
    return decorator


def corrgroups(
    num_features: int = 60, n_points: int = 1_000
) -> tuple[np.ndarray, np.ndarray]:
    """Correlated Groups Dataset from shap.

    This dataset is adapted from the `corrgroups60` dataset in the `shap` library.
    https://github.com/shap/shap/blob/83140b120d66e24377d5c8d59186f3a815cb4304/shap/datasets.py
    Original copyright by Scott Lundberg (2018).

    Synthetic datasets consisting of features with tight correlations
    among distinct groups of features.
    The number of features can be chosen arbitrarily.

    Parameters
    ----------
    num_features : int, optional
        Number of features to generate. Default is 60.
    n_points : int, optional
        Number of data points to generate. Default is 1,000.

    Returns
    -------
    x : np.ndarray
        The feature data matrix
    y : np.ndarray
        The target variables

    Notes
    -----
    - The dataset is generated with known correlations among distinct groups of features.
    - Each feature is a unit variance Gaussian random variable centred around 0.
    - The labels are generated based on a linear function of the features with added random noise.

    Examples
    --------
    .. code-block:: python

        data, target = corrgroups60()

    """
    # set a constant seed
    old_seed = np.random.seed()
    np.random.seed(0)

    # generate dataset with known correlation
    n, m = n_points, num_features

    # set one coefficient from each group of 3 to 1
    beta = np.zeros(m)
    beta[0 : n // 2 : 3] = 1

    # build a correlation matrix with groups of 3 tightly correlated features
    c = np.eye(m)
    for i in range(0, m // 2, 3):
        c[i, i + 1] = c[i + 1, i] = 0.99
        c[i, i + 2] = c[i + 2, i] = 0.99
        c[i + 1, i + 2] = c[i + 2, i + 1] = 0.99

    # Make sure the sample correlation is a perfect match
    x_start = np.random.randn(n, m)
    x_centered = x_start - x_start.mean(0)
    sigma = np.matmul(x_centered.T, x_centered) / x_centered.shape[0]
    w = np.linalg.cholesky(np.linalg.inv(sigma)).T
    x_white = np.matmul(x_centered, w.T)
    assert (
        np.linalg.norm(np.corrcoef(np.matmul(x_centered, w.T).T) - np.eye(m)) < 1e-6
    )  # ensure this decorrelates the data

    # create the final data
    x = np.matmul(x_white, np.linalg.cholesky(c).T)
    y = np.matmul(x, beta) + np.random.randn(n) * 1e-2

    # restore the previous numpy random seed
    np.random.seed(old_seed)

    return x, y


def independentlinear(
    num_features: int = 60, n_points: int = 1_000
) -> tuple[np.ndarray, np.ndarray]:
    """Independent Linear Dataset from shap.

    This dataset is adapted from the `independentlinear60` dataset in the `shap` library.
    https://github.com/shap/shap/blob/83140b120d66e24377d5c8d59186f3a815cb4304/shap/datasets.py
    Original copyright by Scott Lundberg (2018).

    A synthetic dataset with an arbitrary number of features.

    Parameters
    ----------
    num_features : int, optional
        Number of features to generate. Default is 60.
    n_points : int, optional
        Number of data points to generate. Default is 1,000.

    Returns
    -------
    x : np.ndarray
        The feature data matrix
    y : np.ndarray
        The target variables

    Notes
    -----
    - Each feature is a unit variance Gaussian random variable centred around 0.
    - The labels are generated based on a linear function of the features with added random noise.

    Examples
    --------
    .. code-block:: python

        features, labels = shap.datasets.independentlinear60()

    """
    # set a constant seed
    old_seed = np.random.seed()
    np.random.seed(0)

    # generate dataset with known correlation
    n, m = n_points, num_features

    # set one coefficient from each group of 3 to 1
    beta = np.zeros(m)
    beta[0 : n // 2 : 3] = 1

    # Make sure the sample correlation is a perfect match
    x_start = np.random.randn(n, m)
    x = x_start - x_start.mean(0)
    y = np.matmul(x, beta) + np.random.randn(n) * 1e-2

    # restore the previous numpy random seed
    np.random.seed(old_seed)

    return x, y


def german_credit() -> tuple[np.ndarray, np.ndarray]:
    """German Credit Dataset from the UCI ML Repository."""
    dataset = fetch_ucirepo(id=144)

    x = dataset.data.features
    x = categorical_to_int(x)
    x = x.to_numpy().astype(np.float32)

    y = dataset.data.targets
    y = (y > 1).to_numpy().astype(np.bool_)

    return x, y


def german() -> tuple[np.ndarray, np.ndarray]:
    return german_credit()


def obesity() -> tuple[np.ndarray, np.ndarray]:
    """Obesity Levels Based On Eating Habits and Physical Condition Dataset
    from the UCI ML Repository."""
    dataset = fetch_ucirepo(id=544)

    x = dataset.data.features
    x.loc[:, "Gender"] = x["Gender"].map({"Male": 0, "Female": 1})
    x.loc[:, "family_history_with_overweight"] = x["family_history_with_overweight"].map(
        {"yes": 1, "no": 0}
    )
    x.loc[:, "FAVC"] = x["FAVC"].map({"yes": 1, "no": 0})
    x.loc[:, "CAEC"] = x["CAEC"].map({"Sometimes": 1, "Frequently": 2, "Always": 3, "no": 0})
    x.loc[:, "SMOKE"] = x["SMOKE"].map({"yes": 1, "no": 0})
    x.loc[:, "SCC"] = x["SCC"].map({"yes": 1, "no": 0})
    x.loc[:, "CALC"] = x["CALC"].map({"Sometimes": 1, "Frequently": 2, "Always": 3, "no": 0})
    x.loc[:, "MTRANS"] = x["MTRANS"].map(
        {
            "Automobile": 0,
            "Public_Transportation": 1,
            "Motorbike": 2,
            "Bike": 3,
            "Walking": 4,
        }
    )
    x = x.to_numpy().astype(np.float32)

    y = dataset.data.targets
    y = y["NObeyesdad"].map(
        {
            "Insufficient_Weight": 0,
            "Normal_Weight": 1,
            "Overweight_Level_I": 2,
            "Overweight_Level_II": 3,
            "Obesity_Type_I": 4,
            "Obesity_Type_II": 5,
            "Obesity_Type_III": 6,
        }
    )
    y = y.to_numpy().astype(np.int32)

    return x, y


def vehicles() -> tuple[np.ndarray, np.ndarray]:
    """Vehicle class from shape dataset from the UCI ML Repository."""
    dataset = fetch_ucirepo(id=149)

    x = dataset.data.features
    x.loc[:, "COMPACTNESS"] = x["COMPACTNESS"].fillna(x["COMPACTNESS"].mean())
    x = x.to_numpy().astype(np.float32)

    y = dataset.data.targets
    y = y["class"].map(
        {
            "van": 0,
            "saab": 1,
            "bus": 2,
            "opel": 3,
            "204": 0,  # there is just one sample of this class
        }
    )
    y = y.to_numpy().astype(np.int32)

    return x, y


def parkinsons() -> tuple[np.ndarray, np.ndarray]:
    """Parkinsons Telemonitoring dataset from the UCI ML Repository."""
    dataset = fetch_ucirepo(id=189)

    x = dataset.data.features
    x = x.to_numpy().astype(np.float32)

    y = dataset.data.targets
    y = y.to_numpy().astype(np.float32)

    return x, y

@save_dataset("cdc_diabetes")
def cdc_diabetes() -> tuple[np.ndarray, np.ndarray]:
    """Appartment for Rent Dataset from the UCI ML Repository."""
    dataset = fetch_ucirepo(id=891)

    x = dataset.data.features
    x = x.to_numpy().astype(np.float32)

    y = dataset.data.targets
    y = y.to_numpy().astype(np.bool_)

    return x, y


def mushroom() -> tuple[np.ndarray, np.ndarray]:
    """Mushroom Dataset from the UCI ML Repository.

    Predict whether a mushroom is edible or poisonous.
    """
    dataset = fetch_ucirepo(id=73)

    x = dataset.data.features
    x = categorical_to_int(x)
    x = x.to_numpy().astype(np.float32)

    y = dataset.data.targets
    y = (y == "p").to_numpy().astype(np.bool_)

    return x, y


def default() -> tuple[np.ndarray, np.ndarray]:
    """Default of Credit Card Clients Dataset from the UCI ML Repository."""
    dataset = fetch_ucirepo(id=350)

    x = dataset.data.features
    x = categorical_to_int(x)
    x = x.to_numpy().astype(np.float32)

    y = dataset.data.targets
    y = y.to_numpy().astype(np.bool_)

    return x, y


def automobile() -> tuple[np.ndarray, np.ndarray]:
    """Automobile Dataset from the UCI ML Repository."""
    dataset = fetch_ucirepo(id=10)

    x = dataset.data.features
    x = categorical_to_int(x)
    x = x.to_numpy().astype(np.float32)
    x = np.nan_to_num(x)

    y = dataset.data.targets
    y = y["symboling"]
    y = y.to_numpy().astype(np.float32)  # make this a regression task

    return x, y


def steel() -> tuple[np.ndarray, np.ndarray]:
    """Steel Plates Faults Dataset from the UCI ML Repository."""
    dataset = fetch_ucirepo(id=198)

    x = dataset.data.features
    x = x.to_numpy().astype(np.float32)

    y = dataset.data.targets
    y = y.to_numpy().astype(np.float32)  # make this a regression task

    return x, y

@save_dataset("uci_appliances_energy_prediction")
def appliances() -> tuple[np.ndarray, np.ndarray]:
    """Appliances Energy Prediction Dataset from the UCI ML Repository."""
    dataset = fetch_ucirepo(id=374)

    x = dataset.data.features
    x.loc[:, "date"] = x["date"].str[-8:].str.replace(":", "").astype(int)
    x = x.to_numpy().astype(np.float32)

    y = dataset.data.targets
    y = y.to_numpy().astype(np.float32)  # make this a regression task

    return x, y


def hepatitis() -> tuple[np.ndarray, np.ndarray]:
    """Hepatitis C Virus (HCV) for Egyptian patients Dataset from
    the UCI ML Repository.
    """
    dataset = fetch_ucirepo(id=503)

    x = dataset.data.features
    x = categorical_to_int(x)
    x = x.to_numpy().astype(np.float32)

    y = dataset.data.targets
    y = y.iloc[:, 0] - 1
    y = y.to_numpy().astype(np.int32)

    return x, y


def breast_cancer() -> tuple[np.ndarray, np.ndarray]:
    """Breast Cancer Dataset from the UCI ML Repository."""
    dataset = fetch_ucirepo(id=17)

    x = dataset.data.features
    x = x.to_numpy().astype(np.float32)

    y = dataset.data.targets
    y = (y == "M").to_numpy().astype(np.bool_)

    return x, y


def infrared_temperature() -> tuple[np.ndarray, np.ndarray]:
    """Infrared Thermography Temperature Dataset from the UCI ML Repository."""
    dataset = fetch_ucirepo(id=925)

    x = dataset.data.features
    x = categorical_to_int(x)
    x = x.to_numpy().astype(np.float32)
    x = np.nan_to_num(x)

    y = dataset.data.targets
    y = y.to_numpy().astype(np.float32)

    return x, y



def ionosphere() -> tuple[np.ndarray, np.ndarray]:
    """Ionosphere Dataset from the UCI ML Repository."""
    dataset = fetch_ucirepo(id=52)

    x = dataset.data.features
    x = x.to_numpy().astype(np.float32)

    y = dataset.data.targets
    y = (y == "g").to_numpy().astype(np.bool_)

    return x, y


def dropout() -> tuple[np.ndarray, np.ndarray]:
    """Student Dropout and Academic Success Dataset from the UCI ML Repository."""
    dataset = fetch_ucirepo(id=697)

    x = dataset.data.features
    x = x.to_numpy().astype(np.float32)

    y = dataset.data.targets
    y = (y == "Dropout").to_numpy().astype(np.bool_)  # make binary classification task

    return x, y


def annealing() -> tuple[np.ndarray, np.ndarray]:
    """Annealing Dataset from the UCI ML Repository."""
    dataset = fetch_ucirepo(id=3)

    x = dataset.data.features
    x = categorical_to_int(x)
    x = x.to_numpy().astype(np.float32)
    x = np.nan_to_num(x)

    y = dataset.data.targets
    y = (y != "3").to_numpy().astype(np.bool_)  # make binary classification task

    return x, y


@save_dataset("uci_census_income_kdd")
def support2() -> tuple[np.ndarray, np.ndarray]:
    """Support2 Dataset from the UCI ML Repository."""
    dataset = fetch_ucirepo(id=880)

    x = dataset.data.features
    x = categorical_to_int(x)
    x = x.to_numpy().astype(np.float32)
    x = np.nan_to_num(x)

    y = dataset.data.targets
    y = y["death"]
    y = y.to_numpy().astype(np.bool_)

    return x, y



@save_dataset("uci_diabetes130")
def diabetes130() -> tuple[np.ndarray, np.ndarray]:
    """Diabetes j130 Dataset from the UCI ML Repository."""
    dataset = fetch_ucirepo(id=296)

    x = dataset.data.features
    x = categorical_to_int(x)
    x = x.to_numpy().astype(np.float32)
    x = np.nan_to_num(x)

    y = dataset.data.targets
    y = (y != "NO").to_numpy().astype(np.bool_)  # make binary classification task

    return x, y




@save_dataset("uci_covertype")
def covertype() -> tuple[np.ndarray, np.ndarray]:
    """Covertype Dataset from the UCI ML Repository."""
    dataset = fetch_ucirepo(id=31)

    x = dataset.data.features
    x = x.to_numpy().astype(np.float32)

    y = dataset.data.targets
    y = categorical_to_int(y) - 1
    y = y.to_numpy().astype(np.int32).squeeze()

    return x, y
