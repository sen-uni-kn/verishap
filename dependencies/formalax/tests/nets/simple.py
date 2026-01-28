#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, nn


def simple_relu(x):
    """``nn.relu`` has a custom jvp rule and is jit'ed."""
    zero = jnp.zeros_like(x)
    return lax.max(x, zero)


def affine_layer(x, weight, bias):
    return jnp.dot(x, weight) + bias


def conv_layer(x, kernel, bias):
    return lax.conv(x, kernel, window_strides=(1, 1), padding="VALID") + bias


def shallow_nn(x, w1, b1, w2, b2):
    x = x @ jnp.transpose(w1) + b1
    x = nn.relu(x)
    return x @ jnp.transpose(w2) + b2


shallow_nn_jit = jax.jit(shallow_nn)


@partial(jax.vmap, in_axes=(0, None, None, None, None))
def shallow_vmapped(x, w1, b1, w2, b2):
    x = w1 @ x + b1
    x = nn.relu(x)
    return w2 @ x + b2


shallow_vmapped_jit = jax.jit(shallow_vmapped)


def linear_nn(x, w1, b1, w2, b2):
    x = x @ jnp.transpose(w1) + b1
    return x @ jnp.transpose(w2) + b2


linear_nn_jit = jax.jit(linear_nn)


def tanh_two_layer_nn(x, w1, b1, w2, b2, w3, b3):
    for w, b in [(w1, b1), (w2, b2)]:
        x = jnp.tanh(jnp.dot(x, w) + b)
    x = jnp.dot(x, w3) + b3
    return x


def vmapped_avg_of_vmapped(x, w1, b1, w2, b2):
    model = jax.vmap(shallow_vmapped, in_axes=(0, None, None, None, None))
    y = model(x, w1, b1, w2, b2)
    return jnp.mean(y, axis=1)
