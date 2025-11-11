# Copyright 2025 David Boetius
from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Real


def baseline_value(
    model: Callable[[Real[Array, " b *n"]], Real[Array, " b m"]],
    sample: Real[Array, " *n"],
    baseline: Real[Array, " *n"],
    output: int | None = None,
) -> Callable[[Real[Array, " *n"], Bool[Array, " b *n"]], Real[Array, " b"]]:
    """The baseline SHAP value function.

    Args:
        model: The model to evaluate.
        sample: The sample to explain.
        baseline: The baseline value.
        output: The index of the output to explain.
    """
    baseline = jnp.expand_dims(baseline, axis=0)
    x = jnp.expand_dims(sample, axis=0)

    def value(coalitions: Bool[Array, " b *n"]):
        z = coalitions * x + (1 - coalitions) * baseline
        if output is not None:
            return model(z)[..., output]
        else:
            return model(z)

    return value


def superfeature_baseline_value(
    model: Callable[[Real[Array, " b *n"]], Real[Array, " b m"]],
    sample: Real[Array, " *n"],
    baseline: Real[Array, " *n"],
    masks: Bool[Array, " sf *n"],
    output: int | None = None,
) -> Callable[[Real[Array, " sf"], Bool[Array, " b sf"]], Real[Array, " b"]]:
    """The baseline SHAP value function for superfeatures.

    Args:
        model: The model to evaluate.
        sample: The sample to explain.
        baseline: The baseline value.
        masks: The masks for the superfeatures in the input.
        output: The index of the output to explain.

    Returns:
        A value function that reads a boolean vector indicating which
        superfeatures are included and returns a scalar value.
    """
    in_ndim = sample.ndim
    in_dims = tuple(range(-in_ndim, 0))

    def value(coalitions: Bool[Array, " b sf"]):
        sf_coali: Bool[Array, " b *n"] = (
            jnp.expand_dims(coalitions, axis=in_dims) * masks
        ).sum(axis=-in_ndim - 1)
        z = sf_coali * sample + (1 - sf_coali) * baseline
        if output is not None:
            return model(z)[..., output]
        else:
            return model(z)

    return value


def marginal_value(
    model: Callable[[Real[Array, " b *n"]], Real[Array, " b m"]],
    sample: Real[Array, " *n"],
    background_data: Real[Array, " d *n"],
    output: int | None = None,
) -> Callable[[Real[Array, " *n"], Bool[Array, " b *n"]], Real[Array, " b"]]:
    """The Marginal SHAP value function.

    Args:
        model: The model to evaluate.
        sample: The sample to explain.
        background_data: The background data, for example,
            samples from the training data.
        output: The index of the output to explain.
    """
    background: Real[Array, " 1 d *n"] = jnp.expand_dims(background_data, axis=0)
    # add an extra batch axis for the background data
    model: Callable[[Real[Array, " b d *n"]], Real[Array, " b d m"]] = jax.vmap(
        model, in_axes=1, out_axes=1, axis_name="background"
    )
    x: Real[Array, " 1 1 *n"] = jnp.expand_dims(sample, axis=(0, 1))

    def value(coalitions: Bool[Array, " b *n"]):
        coalitions: Real[Array, " b 1 *n"] = jnp.expand_dims(coalitions, axis=1)
        z: Real[Array, " b d *n"] = coalitions * x + (1 - coalitions) * background
        out = jnp.mean(model(z), axis=1)
        if output is not None:
            return out[..., output]
        else:
            return out

    return value


def superfeature_marginal_value(
    model: Callable[[Real[Array, " b *n"]], Real[Array, " b m"]],
    sample: Real[Array, " *n"],
    background_data: Real[Array, " d *n"],
    masks: Bool[Array, " sf *n"],
    output: int | None = None,
) -> Callable[[Real[Array, " sf"], Bool[Array, " b sf"]], Real[Array, " b"]]:
    """The marginal SHAP value function for superfeatures.

    Args:
        model: The model to evaluate.
        sample: The sample to explain.
        background_data: The background data, for example,
            samples from the training data.
        masks: The masks for the superfeatures in the input.
        output: The index of the output to explain.

    Returns:
        A value function that reads a boolean vector indicating which
        superfeatures are included and returns a scalar value.
    """
    in_ndim = sample.ndim
    in_dims = tuple(range(-in_ndim, 0))

    background: Real[Array, " 1 d *n"] = jnp.expand_dims(background_data, axis=0)
    # add an extra batch axis for the background data
    model: Callable[[Real[Array, " b d *n"]], Real[Array, " b d m"]] = jax.vmap(
        model, in_axes=1, out_axes=1, axis_name="background"
    )
    x: Real[Array, " 1 1 *n"] = jnp.expand_dims(sample, axis=(0, 1))

    def value(coalitions: Bool[Array, " b sf"]):
        sf_coali: Bool[Array, " b sf *n"] = (
            jnp.expand_dims(coalitions, axis=in_dims) * masks
        )
        sf_coali: Bool[Array, " b *n"] = sf_coali.sum(axis=1)
        sf_coali: Real[Array, " b 1 *n"] = jnp.expand_dims(sf_coali, axis=1)
        z: Real[Array, " b d *n"] = sf_coali * x + (1 - sf_coali) * background
        out = jnp.mean(model(z), axis=1)
        if output is not None:
            return out[..., output]
        else:
            return out

    return value
