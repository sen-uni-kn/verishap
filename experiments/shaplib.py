#  Copyright (c) 2025. The Formalax Authors.
#  Licensed under the MIT license.
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import shap
from jaxtyping import Array, Bool, Real


def model_wrapper(model):
    """Wraps a model to handle np.ndarray inputs."""
    model = jax.jit(model)

    def wrapper(x: np.ndarray) -> np.ndarray:
        x = jnp.asarray(x)
        y = model(x)
        return np.asarray(y)

    return wrapper


def superfeature_masker(
    sample: Real[Array, " *n"],
    background_data: Real[Array, " d *n"],
    superfeature_masks: Bool[Array, " sf *n"],
) -> Callable[[Bool[Array, " sf *n"], Real[Array, " *n"]], Real[Array, " *n"]]:
    in_ndim = sample.ndim
    in_dims = tuple(range(-in_ndim, 0))
    sample: Real[Array, " 1 *n"] = jnp.expand_dims(sample, axis=0)

    def masker(mask: Bool[Array, " sf"], x: Real[Array, " sf"]) -> Real[Array, " d *n"]:
        sf_mask: Bool[Array, " sf *n"] = (
            jnp.expand_dims(mask, axis=in_dims) * superfeature_masks
        )
        sf_mask: Bool[Array, " *n"] = sf_mask.sum(axis=0)
        sf_mask: Real[Array, " 1 *n"] = jnp.expand_dims(sf_mask, axis=0)
        z: Real[Array, " d *n"] = sf_mask * sample + (1 - sf_mask) * background_data
        return z

    return masker


def _preprocess_array(x: Real[Array, " *n"]) -> Real[Array, " d *n"]:
    x = np.asarray(x)
    x = np.atleast_2d(x)
    return x


def _preprocess_masker(masker: Callable | Real[Array, " *n"] | Real[Array, " d *n"]) -> Real[Array, " d *n"]:
    if not isinstance(masker, Callable):
        masker = _preprocess_array(masker)
    return masker


def exact_shap(
    model: Callable[[Real[Array, " b *n"]], Real[Array, " b m"]],
    masker: Callable | Real[Array, " *n"] | Real[Array, " d *n"],
    x: Real[Array, " *n"],
    num_samples: int | None = None,
    silent: bool = False,
) -> Real[Array, " b *n"]:
    """The Exact SHAP explainer from the `shap` library.

    Computes exact SHAP values using enumeration.
    """
    model = model_wrapper(model)
    x = _preprocess_array(x)
    masker = _preprocess_masker(masker)

    explainer = shap.ExactExplainer(model, masker=masker)
    explanation = explainer(x, max_evals=num_samples, silent=silent)
    return explanation.values[0]


def kernel_shap(model, baseline, x, num_samples=1024, silent=False):
    """The Kernel SHAP explainer from the `shap` library."""
    if isinstance(baseline, Callable):
        raise ValueError("KernelSHAP does not support callable maskers.")

    model = model_wrapper(model)
    baseline, x = np.asarray(baseline), np.asarray(x)
    baseline = np.atleast_2d(baseline)

    explainer = shap.KernelExplainer(model, data=baseline)
    shap_values = explainer.shap_values(
        x, nsamples=num_samples, silent=silent, l1_reg=False
    )
    return shap_values


def permutation_shap(model, masker, x, num_samples=1024, silent=False):
    """The Permutation SHAP explainer from the `shap` library."""
    model = model_wrapper(model)
    x = _preprocess_array(x)
    masker = _preprocess_masker(masker)

    x = x.astype("float64")
    num_features = x.shape[-1]
    num_permutations = num_samples // num_features

    explainer = shap.PermutationExplainer(model, masker=masker)
    shap_values = explainer.shap_values(
        x, npermutations=num_permutations, silent=silent
    )
    return shap_values[0]


def sampling_shap(model, masker, x, num_samples=1024, silent=False):
    """The Sampling SHAP explainer from the `shap` library."""
    model = model_wrapper(model)
    x = _preprocess_array(x)
    masker = _preprocess_masker(masker)

    explainer = shap.SamplingExplainer(model, masker=masker)
    shap_values = explainer.shap_values(x, nsamples=num_samples, silent=silent)
    return shap_values
