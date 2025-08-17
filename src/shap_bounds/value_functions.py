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
        return model(z)[..., output]

    return value
