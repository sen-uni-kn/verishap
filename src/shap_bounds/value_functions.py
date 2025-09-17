# Copyright 2025 David Boetius
from typing import Callable

import jax.numpy as jnp
from jaxtyping import Array, Bool, Real


def baseline_value(
    model: Callable[[Real[Array, " b *n"]], Real[Array, " b m"]],
    baseline: Real[Array, " *n"],
    output: int | None = None,
) -> Callable[[Real[Array, " *n"], Bool[Array, " b *n"]], Real[Array, " b"]]:
    """The baseline SHAP value function.

    Args:
        model: The model to evaluate.
        baseline: The baseline value.
        output: The index of the output to explain.
    """
    baseline = jnp.expand_dims(baseline, axis=0)

    def value(x: Real[Array, " *n"], coalitions: Bool[Array, " b *n"]):
        x = jnp.expand_dims(x, axis=0)
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
        baseline: The baseline value.
        masks: The masks for the superfeatures in the input.
        output: The index of the output to explain.

    Returns:
        A value function that reads a boolean vector indicating which superfeatures are included
        and returns a scalar value.
    """
    in_dims = tuple(range(-sample.ndim, 0))

    def value(_: Real[Array, " sf"], coalitions: Bool[Array, " b sf"]):
        sf_coali: Bool[Array, " b *n"] = (jnp.expand_dims(coalitions, axis=in_dims) * masks).sum(axis=1)
        z = sf_coali * sample + (1 - sf_coali) * baseline
        if output is not None:
            return model(z)[..., output]
        else:
            return model(z)

    return value
