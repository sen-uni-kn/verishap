#  Copyright (c) 2025. David Boetius.
from typing import Callable

import numpy as np
from jaxtyping import Array, Real

from ..shaplib import model_wrapper
from .lshap import LeverageSHAPEstimator
from .regMSR import LinearMSR, TreeMSR


class Wrapper:
    """Wraps a model or value function to provide a .predict method that handles
    np.ndarray inputs.
    """
    def __init__(self, model, squeeze: bool = True):
        self.wrapper = model_wrapper(model)
        self.squeeze = squeeze

    def predict(self, x: np.ndarray) -> np.ndarray:
        y = self.wrapper(x)
        return y.squeeze() if self.squeeze else y


def run_estimator(
    value_function: Callable[[Real[Array, " b *n"], Real[Array, " *n"]], Real[Array, " d"]],
    base_mask: Real[Array, " *n"],
    explainer_class,
    num_samples: int | None = None,
    seed: int | None = None,
):
    model = Wrapper(value_function, squeeze=True)
    x = np.ones((1, *base_mask.shape), dtype=base_mask.dtype)
    baseline = np.zeros((1, *base_mask.shape), dtype=base_mask.dtype)

    explainer = explainer_class(model, baseline, weighting="shapley", seed=seed)
    return explainer.explain(x, num_samples).reshape(-1, 1)


def linear_msr(
    value_function: Callable[[Real[Array, " b *n"], Real[Array, " *n"]], Real[Array, " d"]],
    base_mask: Real[Array, " *n"],
    num_samples: int | None = None,
    seed: int | None = None,
):
    return run_estimator(value_function, base_mask, LinearMSR, num_samples, seed)


def tree_msr(
    value_function: Callable[[Real[Array, " b *n"], Real[Array, " *n"]], Real[Array, " d"]],
    base_mask: Real[Array, " *n"],
    num_samples: int | None = None,
    seed: int | None = None,
):
    return run_estimator(value_function, base_mask, TreeMSR, num_samples, seed)


def leverage_shap(
    value_function: Callable[[Real[Array, " b *n"], Real[Array, " *n"]], Real[Array, " d"]],
    base_mask: Real[Array, " *n"],
    num_samples: int | None = None,
    seed: int | None = None,
):
    return run_estimator(value_function, base_mask, LeverageSHAPEstimator, num_samples, seed)
