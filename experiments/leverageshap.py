#  Copyright (c) 2025. David Boetius.
from typing import Callable

import leverageshap as ls
import numpy as np
from jaxtyping import Array, Real

from .shaplib import _preprocess_array, _preprocess_masker, model_wrapper


class ModelWrapper:
    """Wraps a model to provide a .predict method that handles
    np.ndarray inputs.
    """
    def __init__(self, model, batch_size: int = 16384):
        self.wrapper = model_wrapper(model, batch_size)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.wrapper(x)


def leverage_shap(
    model: Callable[[Real[Array, " b *n"]], Real[Array, " b m"]],
    masker: Callable | Real[Array, " *n"] | Real[Array, " d *n"],
    x: Real[Array, " *n"],
    num_samples: int | None = None,
    batch_size: int = 16384,
):
    model = ModelWrapper(model, batch_size)
    x = _preprocess_array(x)
    masker = _preprocess_masker(masker)

    return ls.leverage_shap(masker, x, model, num_samples)
