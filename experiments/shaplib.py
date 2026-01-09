#  Copyright (c) 2025. David Boetius.
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import shap
from jaxtyping import Array, Bool, Real


def model_wrapper(model, batch_size: int = 16384):
    """Wraps a model to handle np.ndarray inputs.

    Divides the input into batches of size `batch_size` to handle large inputs.
    """
    model = jax.jit(model)

    def wrapper(x: np.ndarray) -> np.ndarray:
        ys = []
        for i in range(0, x.shape[0], batch_size):
            batch = x[i:i+batch_size]
            y = model(batch)
            ys.append(np.asarray(y))
        return np.concatenate(ys, axis=0)

    return wrapper


def superfeature_masker(
    sample: Real[Array, " *n"],
    background_data: Real[Array, " d *n"],
    superfeature_masks: Bool[Array, " sf *n"],
) -> Callable[[Bool[Array, " sf *n"], Real[Array, " *n"]], Real[Array, " *n"]]:
    in_ndim = sample.ndim
    in_dims = tuple(range(-in_ndim, 0))
    sample: Real[Array, " 1 *n"] = jnp.expand_dims(sample, axis=0)

    def masker(mask: Bool[Array, " b sf"], x: Real[Array, " sf"]) -> Real[Array, " b d *n"]:
        # mask batch dimension is optional
        n_batch = len(mask.shape) - 1
        sf_mask: Bool[Array, " b sf *n"] = (
            jnp.expand_dims(mask, axis=in_dims) * superfeature_masks
        )
        sf_mask: Bool[Array, " b *n"] = sf_mask.sum(axis=n_batch)
        sf_mask: Real[Array, " b 1 *n"] = jnp.expand_dims(sf_mask, axis=n_batch)
        z: Real[Array, " b d *n"] = sf_mask * sample + (1 - sf_mask) * background_data
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
    batch_size: int = 16384,
) -> Real[Array, " b *n"]:
    """The Exact SHAP explainer from the `shap` library.

    Computes exact SHAP values using enumeration.
    """
    model = model_wrapper(model, batch_size)
    x = _preprocess_array(x)
    masker = _preprocess_masker(masker)

    explainer = shap.ExactExplainer(model, masker=masker)
    explanation = explainer(x, max_evals=num_samples, silent=silent)
    shap_values = explanation.values
    shap_values = np.moveaxis(shap_values, 0, -1)  # to: n_features x n_outputs
    return shap_values


def kernel_shap(model, baseline, x, num_samples=1024, silent=False, batch_size: int = 16384):
    """The Kernel SHAP explainer from the `shap` library."""
    if isinstance(baseline, Callable):
        raise ValueError("KernelSHAP does not support callable maskers.")

    model = model_wrapper(model, batch_size)
    x = np.asarray(x)
    baseline = _preprocess_array(baseline)

    explainer = shap.KernelExplainer(model, data=baseline)
    shap_values = explainer.shap_values(
        x, nsamples=num_samples, silent=silent, l1_reg=False
    )
    return shap_values


def permutation_shap(model, masker, x, num_samples=1024, silent=False, batch_size: int = 16384):
    """The Permutation SHAP explainer from the `shap` library."""
    model = model_wrapper(model, batch_size)
    x = _preprocess_array(x)
    masker = _preprocess_masker(masker)

    x = x.astype("float64")
    num_features = x.shape[-1]
    num_permutations = num_samples // num_features

    explainer = shap.PermutationExplainer(model, masker=masker)
    shap_values = explainer.shap_values(
        x, npermutations=num_permutations, silent=silent
    )
    shap_values = np.moveaxis(shap_values, 0, -1)  # to: n_features x n_outputs
    return shap_values


def sampling_shap(model, baseline, x, num_samples=1024, silent=False, batch_size: int = 16384):
    """The Sampling SHAP explainer from the `shap` library."""
    if isinstance(baseline, Callable):
        raise ValueError("SamplingSHAP does not support callable baselines.")

    model = model_wrapper(model, batch_size)
    x = _preprocess_array(x)
    baseline = _preprocess_array(baseline)

    explainer = shap.SamplingExplainer(model, data=baseline)
    shap_values = explainer.shap_values(x, nsamples=num_samples, silent=silent)
    shap_values = np.moveaxis(shap_values, 0, -1)  # to: n_features x n_outputs
    return shap_values
