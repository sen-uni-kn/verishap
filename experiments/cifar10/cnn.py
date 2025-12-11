# Copyright 2025 David Boetius
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree


class CNN(eqx.Module):
    """A simple CNN model for classifying CIFAR10 digits.

    Adapted from https://docs.kidger.site/equinox/examples/mnist/
    """

    layers: list

    def __init__(self, key):
        key1, key2, key3, key4 = jax.random.split(key, 4)
        self.layers = [
            partial(jnp.reshape, shape=(3, 32, 32)),
            eqx.nn.Conv2d(3, 4, kernel_size=5, padding=2, key=key1, use_bias=False),
            eqx.nn.AvgPool2d(kernel_size=2, stride=2),
            eqx.nn.BatchNorm(4, axis_name="batch", mode="batch"),
            jax.nn.relu,
            eqx.nn.Conv2d(4, 8, kernel_size=5, padding=2, key=key2, use_bias=False),
            eqx.nn.AvgPool2d(kernel_size=2, stride=2),
            eqx.nn.BatchNorm(8, axis_name="batch", mode="batch"),
            jax.nn.relu,
            jnp.ravel,
            eqx.nn.Linear(512, 64, key=key3),
            jax.nn.relu,
            eqx.nn.Linear(64, 1, key=key4),
        ]

    def __call__(
        self, x: Float[Array, "3 32 32"], state: PyTree
    ) -> tuple[Float[Array, ""], PyTree]:
        for layer in self.layers:
            if isinstance(layer, eqx.nn.BatchNorm):
                x, state = layer(x, state)
            else:
                x = layer(x)
        x = x.squeeze(axis=-1)
        return x, state
