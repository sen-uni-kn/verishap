#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
from functools import partial
from pathlib import Path
from typing import Callable, Literal, Sequence

import jax.numpy as jnp
import jax.random
import jax.tree
import numpy as np
import pytest
from flax import linen as nn
from jaxtyping import PyTree


class FlaxConv(nn.Module):
    """A flexible Convolutional Neural Network model definition.

    Attributes:
        input_shape: The shape of the input data.
        model_arch: A sequence of tuples, where each tuple represents a layer
            in the model. The first element of each tuple is a string
            representing the layer type, and the remaining elements are the
            layer parameters. The supported layer types are:
                - "conv": A convolutional layer with are square kernel.
                    The second tuple element is the number of filters.
                    The third tuple element is the kernel size.
                    The fourth tuple element is a boolean indicating whether
                    the convolutional layer has a bias term.
                - "batch_norm": A batch normalization layer.
                - "relu": A ReLU layer.
                - "avg_pool": An average pooling layer with a square window.
                    The first tuple element is the window size.
                    The second tuple element is the stride.
                - "dense": A dense layer.
                    The first tuple element is the number of features.
                    The second tuple element is a boolean indicating whether
                    the dense layer has a bias term.
            Flattening layers are added automatically.
            Implicitly, the final layer is always a dense layer with the number
            of classes as features.
            You do not have to include this layer.
        num_classes: The number of output classes.
    """

    input_shape: tuple[int, ...]
    model_arch: Sequence[
        tuple[Literal["relu"] | Literal["batch_norm"]]
        | tuple[Literal["conv"], int, int, bool]
        | tuple[Literal["avg_pool"], int, int]
        | tuple[Literal["dense"], int, bool]
    ]
    num_classes: int

    @nn.compact
    def __call__(self, x, training=False):
        x = x.reshape(-1, *self.input_shape)

        prev_is_image = len(self.input_shape) > 1
        for layer in self.model_arch:
            match layer:
                case ("conv", filters, kernel_size, has_bias):
                    x = nn.Conv(
                        features=filters,
                        kernel_size=(kernel_size, kernel_size),
                        use_bias=has_bias,
                    )(x)
                case ("batch_norm",):
                    x = nn.BatchNorm(use_running_average=not training)(x)
                case ("relu",):
                    x = nn.relu(x)
                case ("avg_pool", window_size, stride):
                    x = nn.avg_pool(
                        x,
                        window_shape=(window_size, window_size),
                        strides=(stride, stride),
                    )
                case ("dense", features, has_bias):
                    if prev_is_image:
                        x = x.reshape(x.shape[0], -1)
                        prev_is_image = False
                    x = nn.Dense(features=features, use_bias=has_bias)(x)
        return nn.Dense(features=self.num_classes)(x)


def load_flax_params(
    model: nn.Module,
    params_path: str | Path,
) -> Callable:
    """Loads the parameters of a Flax model from a pickle file.

    Returns:
        A callable applying `model` with the loaded parameters.
    """
    placeholders = model.init(jax.random.PRNGKey(0), jnp.ones(model.input_shape))
    _, pytree = jax.tree.flatten(placeholders)

    with open(params_path, "rb") as f, np.load(f) as params:
        params = [jnp.asarray(p) for p in params.values()]
    params = jax.tree.unflatten(pytree, params)

    if "params" not in params:
        params = {"params": params}
    return partial(model.apply, params)


@pytest.fixture
def emnist_flax_conv(resource_dir) -> Callable:
    model = FlaxConv(
        input_shape=(28, 28, 1),
        model_arch=[
            ("conv", 32, 3, True),
            ("relu",),
            ("avg_pool", 2, 2),
            ("conv", 64, 3, True),
            ("relu",),
            ("avg_pool", 2, 2),
            ("dense", 256, True),
            ("relu",),
        ],
        num_classes=47,
    )
    return load_flax_params(model, resource_dir / "emnist_conv_flax.npz")


@pytest.fixture
def mnist_ibp_training_flax_conv(resource_dir) -> Callable:
    model = FlaxConv(
        input_shape=(28, 28, 1),
        model_arch=[
            ("conv", 32, 3, False),
            ("batch_norm",),
            ("relu",),
            ("avg_pool", 2, 2),
            ("conv", 64, 3, False),
            ("batch_norm",),
            ("relu",),
            ("avg_pool", 2, 2),
            ("dense", 256, False),
            ("batch_norm",),
            ("relu",),
        ],
        num_classes=10,
    )
    return load_flax_params(model, resource_dir / "mnist_conv_flax_ibp_training.npz")


@pytest.fixture
def mnist_flax_mlp() -> Callable:
    input_shape = (28, 28, 1)
    model = FlaxConv(
        input_shape=input_shape,
        model_arch=[
            ("dense", 26 * 26 * 32, False),
            ("relu",),
            ("dense", 13 * 13 * 32, False),
            ("relu",),
            ("dense", 11 * 11 * 64, False),
            ("relu",),
            ("dense", 5 * 5 * 64, False),
            ("relu",),
            ("dense", 256, False),
            ("relu",),
        ],
        num_classes=10,
    )
    rng_key = jax.random.PRNGKey(0)
    params = model.init(rng_key, jnp.ones(input_shape))
    return partial(model.apply, params)


@pytest.fixture
def emnist_flax_mlp(rng) -> PyTree:
    input_shape = (28, 28, 1)
    model = FlaxConv(
        input_shape=input_shape,
        model_arch=[
            ("dense", 26 * 26 * 32, False),
            ("relu",),
            ("dense", 13 * 13 * 32, False),
            ("relu",),
            ("dense", 11 * 11 * 64, False),
            ("relu",),
            ("dense", 5 * 5 * 64, False),
            ("relu",),
            ("dense", 1600, False),
            ("relu",),
            ("dense", 256, False),
            ("relu",),
        ],
        num_classes=47,
    )
    params = model.init(rng, jnp.ones(input_shape))["params"]
    return params
