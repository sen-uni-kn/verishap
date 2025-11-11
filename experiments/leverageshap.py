#  Copyright (c) 2025. David Boetius.
from typing import Callable

import jax
import jax.numpy as jnp
import leverageshap as ls
import numpy as np
from jaxtyping import Array, Real


class ModelWrapper:
    """Wraps a model to provide a .predict method that handles
    np.ndarray inputs.
    """
    def __init__(self, model):
        self.model = jax.jit(model)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = jnp.asarray(x)
        y = self.model(x)
        return np.asarray(y)


def leverage_shap(
    model: Callable[[Real[Array, " b *n"]], Real[Array, " b m"]],
    baseline: Real[Array, " *n"],
    x: Real[Array, " *n"],
    num_samples: int | None = None,
):
    model = ModelWrapper(model)
    x, baseline = np.asarray(x), np.asarray(baseline)
    x, baseline = np.atleast_2d(x), np.atleast_2d(baseline)

    return ls.leverage_shap(baseline, x, model, num_samples)
