# Copyright (c) 2025. The Formalax Authors.
# Licensed under the MIT license.
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class SumOut(eqx.Module):
    """Sums out the input dimension."""

    def __call__(self, x: Float[Array, " n"]) -> Float[Array, "1"]:
        return jnp.sum(x).reshape(1)


class MLP(eqx.Module):
    """Sums out the input dimension."""

    layers: list

    def __init__(self, input_dim, layers, key):
        keys = jax.random.split(key, len(layers) + 1)
        layers = [input_dim] + layers
        modules = []
        for i in range(1, len(layers)):
            modules.append(eqx.nn.Linear(layers[i - 1], layers[i], key=keys[i]))
            modules.append(jax.nn.relu)
        modules.append(eqx.nn.Linear(layers[-1], 1, key=keys[-1]))  # Output layer
        self.layers = modules

    def __call__(self, x: Float[Array, " n"]) -> Float[Array, "1"]:
        for layer in self.layers:
            x = layer(x)
        return x
