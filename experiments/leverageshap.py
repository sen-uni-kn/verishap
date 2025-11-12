#  Copyright (c) 2025. David Boetius.
from typing import Callable

import jax
import jax.numpy as jnp
import leverageshap as ls
import numpy as np
from jaxtyping import Array, Real

from .shaplib import _preprocess_array, _preprocess_masker


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
    masker: Callable | Real[Array, " *n"] | Real[Array, " d *n"],
    x: Real[Array, " *n"],
    num_samples: int | None = None,
):
    model = ModelWrapper(model)
    x = _preprocess_array(x)
    masker = _preprocess_masker(masker)

    return ls.leverage_shap(masker, x, model, num_samples)
