#  Copyright (c) 2025. The Formalax Authors.
#  Licensed under the MIT license.
from typing import Callable

import jax.numpy as jnp
import numpy as np
import shap
from jaxtyping import Array, Real


def model_wrapper(model):
    """Wraps a model to handle np.ndarray inputs."""

    def wrapper(x: np.ndarray) -> np.ndarray:
        x = jnp.asarray(x)
        y = model(x)
        return np.asarray(y)

    return wrapper


def exact_shap(
    model: Callable[[Real[Array, " b *n"]], Real[Array, " b m"]],
    baseline: Real[Array, " *n"],
    x: Real[Array, " *n"],
    num_samples: int | None = None,
    silent: bool = False,
) -> Real[Array, " b *n"]:
    """The Exact SHAP explainer from the `shap` library.

    Computes exact SHAP values using enumeration.
    """
    model = model_wrapper(model)
    baseline, x = np.asarray(baseline), np.asarray(x)
    baseline, x = np.atleast_2d(baseline), np.atleast_2d(x)

    explainer = shap.ExactExplainer(model, masker=baseline)
    explanation = explainer(x, max_evals=num_samples, silent=silent)
    return explanation.values[0]


def kernel_shap(model, baseline, x, num_samples=1024, silent=False):
    """The Kernel SHAP explainer from the `shap` library."""
    model = model_wrapper(model)
    baseline, x = np.asarray(baseline), np.asarray(x)
    baseline = np.atleast_2d(baseline)

    explainer = shap.KernelExplainer(model, data=baseline)
    shap_values = explainer.shap_values(
        x, nsamples=num_samples, silent=silent, l1_reg=False
    )
    return shap_values


def permutation_shap(model, baseline, x, num_samples=1024, silent=False):
    """The Permutation SHAP explainer from the `shap` library."""
    model = model_wrapper(model)
    baseline, x = np.asarray(baseline), np.asarray(x)
    baseline, x = np.atleast_2d(baseline), np.atleast_2d(x)

    x = x.astype("float64")
    num_features = x.shape[-1]
    num_permutations = num_samples // num_features

    explainer = shap.PermutationExplainer(model, masker=baseline)
    shap_values = explainer.shap_values(
        x, npermutations=num_permutations, silent=silent
    )
    return shap_values[0]


def sampling_shap(model, baseline, x, num_samples=1024, silent=False):
    """The Sampling SHAP explainer from the `shap` library."""
    model = model_wrapper(model)
    baseline, x = np.asarray(baseline), np.asarray(x)
    baseline = np.atleast_2d(baseline)

    explainer = shap.SamplingExplainer(model, masker=baseline)
    shap_values = explainer.shap_values(x, nsamples=num_samples, silent=silent)
    return shap_values
