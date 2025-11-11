# Copyright 2025 David Boetius
import numpy as np


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
