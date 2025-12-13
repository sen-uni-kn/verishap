# Copyright 2025 David Boetius
import numpy as np
import shap.datasets

from .. import datasets
from ..datasets import corrgroups, independentlinear


def load_dataset(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """Loads a tabular dataset."""
    if dataset.startswith("corrgroups"):
        num_features = int(dataset[-2:])
        data, targets = corrgroups(num_features)
    elif dataset.startswith("independentlinear"):
        num_features = int(dataset[-2:])
        data, targets = independentlinear(num_features)
    elif hasattr(datasets, dataset):
        data, targets = getattr(datasets, dataset)()
    elif hasattr(shap.datasets, dataset):
        data, targets = getattr(shap.datasets, dataset)()
        data = data.to_numpy().astype(np.float32)
        data = np.nan_to_num(data)
    else:
        raise ValueError(f"Dataset {dataset} not found.")
    return data, targets
