# Copyright 2025 David Boetius
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree


class CNN(eqx.Module):
    """A simple CNN model for classifying MNIST digits.

    Adapted from https://docs.kidger.site/equinox/examples/mnist/
    """
    layers: list

    def __init__(self, key):
        key1, key2, key3, key4 = jax.random.split(key, 4)
        self.layers = [
            partial(jnp.reshape, shape=(1, 28, 28)),
            eqx.nn.Conv2d(1, 4, kernel_size=5, padding=2, key=key1),
            eqx.nn.AvgPool2d(kernel_size=2, stride=2),
            eqx.nn.BatchNorm(4, axis_name="batch", mode="batch"),
            jax.nn.relu,
            eqx.nn.Conv2d(4, 8, kernel_size=5, padding=2, key=key2),
            eqx.nn.AvgPool2d(kernel_size=2, stride=2),
            eqx.nn.BatchNorm(8, axis_name="batch", mode="batch"),
            jax.nn.relu,
            jnp.ravel,
            eqx.nn.Linear(392, 64, key=key3),
            jax.nn.relu,
            eqx.nn.Linear(64, 10, key=key4),
        ]

    def __call__(self, x: Float[Array, "1 28 28"] | Float[Array, "28 28"], state: PyTree) -> Float[Array, "10"]:
        for layer in self.layers:
            if isinstance(layer, eqx.nn.BatchNorm):
                x, state = layer(x, state)
            else:
                x = layer(x)
        return x, state
