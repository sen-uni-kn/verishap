# Copyright 2025 David Boetius
import io
from collections.abc import Sequence
from functools import partial
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import ruamel.yaml as yaml
from jaxtyping import Array, Float, PyTree


class ZScoreNorm(eqx.Module):
    mean: Float[Array, " n"]
    std: Float[Array, " n"]

    def __call__(self, x: Float[Array, " n"]) -> Float[Array, " n"]:
        return (x - self.mean) / self.std


class ZScoreUnnorm(eqx.Module):
    mean: Float[Array, " n"]
    std: Float[Array, " n"]

    def __call__(self, x: Float[Array, " n"]) -> Float[Array, " n"]:
        return x * self.std + self.mean


class MLP(eqx.Module):
    """A Multi-Layer Perceptron model for classifying tabular data."""

    input_norm: ZScoreNorm
    output_norm: ZScoreUnnorm
    layers: list
    input_dim: int
    output_dim: int
    hidden_dim: int
    hidden_layers: int

    def __init__(
        self,
        input_dim,
        output_dim,
        key,
        hidden_dim=32,
        hidden_layers=2,
        input_norm_stats: tuple[Float[Array, " n"], Float[Array, " n"]] | None = None,
        output_norm_stats: tuple[Float[Array, " m"], Float[Array, " m"]] | None = None,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers

        if input_norm_stats is None:
            input_mean, input_std = jnp.zeros(input_dim), jnp.ones(input_dim)
        else:
            input_mean, input_std = input_norm_stats
        self.input_norm = ZScoreNorm(input_mean, input_std)

        if output_norm_stats is None:
            if output_dim == 1:
                output_mean, output_std = jnp.zeros(()), jnp.ones(())
            else:
                output_mean, output_std = jnp.zeros(output_dim), jnp.ones(output_dim)
        else:
            output_mean, output_std = output_norm_stats
        self.output_norm = ZScoreUnnorm(output_mean, output_std)

        keys = jax.random.split(key, hidden_layers + 1)
        layer_sizes = [input_dim] + [hidden_dim] * hidden_layers + [output_dim]
        layers = []
        for i in range(1, len(layer_sizes)):
            layers.append(
                eqx.nn.Linear(layer_sizes[i - 1], layer_sizes[i], key=keys[i])
            )
            layers.append(jax.nn.relu)
        self.layers = layers[:-1]  # remove training ReLU layer

    def __call__(self, x: Float[Array, " n"]) -> Float[Array, " m"]:
        x = self.input_norm(x)
        for layer in self.layers:
            x = layer(x)
        return self.output_norm(x)

    def save(self, file: str, extra_info: dict | None = None):
        if extra_info is None:
            extra_info = {}
        # Input and output norm are saved as arrays by equinox.
        info_dict = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dim": self.hidden_dim,
            "hidden_layers": self.hidden_layers,
        } | extra_info
        with open(file, "wb") as f:
            yaml_str = io.StringIO()
            yaml.YAML().dump(info_dict, yaml_str)
            f.write(yaml_str.getvalue().encode("utf-8"))
            f.write(b"\n---\n")
            eqx.tree_serialise_leaves(f, self)

    @classmethod
    def load(cls, file: str):
        with open(file, "rb") as f:
            yaml_lines = []
            while True:
                line = f.readline().decode("utf-8")
                if line.startswith("---"):
                    break
                yaml_lines.append(line)
            info_dict = yaml.YAML().load(io.StringIO("\n".join(yaml_lines)))
            model = cls(
                info_dict["input_dim"],
                info_dict["output_dim"],
                jax.random.PRNGKey(0),
                info_dict["hidden_dim"],
                info_dict["hidden_layers"],
                input_norm_stats=None,
                output_norm_stats=None,
            )
            model = eqx.tree_deserialise_leaves(f, model)
            return model


# ==============================================================================
# Vision Models
# ==============================================================================


CONV_LAYER_DEFINITION = dict[
    Literal[
        "channels",
        "conv_kernel_size",
        "conv_stride",
        "conv_padding",
        "pool_kernel_size",
        "pool_stride",
        "pool_padding",
    ],
    int,
]
CONV_LAYER_DEFAULTS = {
    "channels": 4,
    "conv_kernel_size": 5,
    "conv_stride": 1,
    "conv_padding": 2,
    "pool_kernel_size": 2,
    "pool_stride": 2,
    "pool_padding": 0,
}


class CNN(eqx.Module):
    """A simple CNN model for classification.

    Adapted from https://docs.kidger.site/equinox/examples/mnist/
    """

    layers: list
    input_shape: tuple[int, ...]
    output_dim: int

    def __init__(
        self,
        input_shape,
        output_dim,
        key,
        conv_layers: Sequence[CONV_LAYER_DEFINITION] = (
            {"channels": 4},
            {"channels": 8},
        ),
        fc_in_sizes: Sequence[int] = (512, 64),
    ):
        self.input_shape = tuple(input_shape)
        self.output_dim = output_dim

        conv_keys, fc_keys = jax.random.split(key, 2)
        conv_keys = jax.random.split(conv_keys, len(conv_layers))
        layers = [partial(jnp.reshape, shape=input_shape)]
        in_channels = input_shape[-3]
        for defn, key in zip(conv_layers, conv_keys, strict=True):
            out_channels = defn.get("channels", CONV_LAYER_DEFAULTS["channels"])
            conv = eqx.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=defn.get(
                    "conv_kernel_size", CONV_LAYER_DEFAULTS["conv_kernel_size"]
                ),
                stride=defn.get("conv_stride", CONV_LAYER_DEFAULTS["conv_stride"]),
                padding=defn.get("conv_padding", CONV_LAYER_DEFAULTS["conv_padding"]),
                use_bias=False,
                key=key,
            )
            pool = eqx.nn.AvgPool2d(
                kernel_size=defn.get(
                    "pool_kernel_size", CONV_LAYER_DEFAULTS["pool_kernel_size"]
                ),
                stride=defn.get("pool_stride", CONV_LAYER_DEFAULTS["pool_stride"]),
                padding=defn.get("pool_padding", CONV_LAYER_DEFAULTS["pool_padding"]),
            )
            norm = eqx.nn.BatchNorm(out_channels, axis_name="batch", mode="batch")
            layers += [conv, pool, norm, jax.nn.relu]
            in_channels = out_channels

        fc_keys = jax.random.split(fc_keys, len(fc_in_sizes))
        layers.append(jnp.ravel)
        fc_out_sizes = tuple(fc_in_sizes[1:]) + (output_dim,)
        for i in range(len(fc_in_sizes)):
            layers += [
                eqx.nn.Linear(fc_in_sizes[i], fc_out_sizes[i], key=fc_keys[i]),
                jax.nn.relu,
            ]
        layers = layers[:-1]  # remove last ReLU layer
        self.layers = tuple(layers)

    def __call__(
        self, x: Float[Array, " c h w"] | Float[Array, " h w"], state: PyTree
    ) -> tuple[Float[Array, ""], PyTree]:
        for layer in self.layers:
            if isinstance(layer, eqx.nn.BatchNorm):
                x, state = layer(x, state)
            else:
                x = layer(x)
        if self.output_dim == 1:
            x = x.squeeze(axis=-1)
        return x, state

    @classmethod
    def save(
        cls, model: "CNN", state: PyTree, file: str, extra_info: dict | None = None
    ):
        conv_layers = []
        fc_in_sizes = []
        prev_conv = {}
        for layer in model.layers:
            if isinstance(layer, eqx.nn.Conv2d):
                prev_conv = {
                    "channels": layer.out_channels,
                    "conv_kernel_size": layer.kernel_size,
                    "conv_stride": layer.stride,
                    "conv_padding": layer.padding,
                }
            elif isinstance(layer, eqx.nn.AvgPool2d):
                assert len(prev_conv) > 0
                prev_conv["pool_kernel_size"] = layer.kernel_size
                prev_conv["pool_stride"] = layer.stride
                prev_conv["pool_padding"] = layer.padding
                conv_layers.append(prev_conv)
                prev_conv = {}
            elif isinstance(layer, eqx.nn.Linear):
                fc_in_sizes.append(layer.in_features)

        if extra_info is None:
            extra_info = {}
        info_dict = {
            "input_shape": model.input_shape,
            "output_dim": model.output_dim,
            "conv_layers": conv_layers,
            "fc_in_sizes": fc_in_sizes,
        } | extra_info
        with open(file, "wb") as f:
            yaml_str = io.StringIO()
            yaml.YAML().dump(info_dict, yaml_str)
            f.write(yaml_str.getvalue().encode("utf-8"))
            f.write(b"\n---\n")
            eqx.tree_serialise_leaves(f, (model, state))

    @classmethod
    def load(cls, file: str) -> tuple["CNN", PyTree]:
        """Loads a CNN model and its state from a file."""
        with open(file, "rb") as f:
            yaml_lines = []
            while True:
                line = f.readline().decode("utf-8")
                if line.startswith("---"):
                    break
                yaml_lines.append(line)
            info_dict = yaml.YAML().load(io.StringIO("\n".join(yaml_lines)))
            model, state = eqx.nn.make_with_state(cls)(
                info_dict["input_shape"],
                info_dict["output_dim"],
                jax.random.PRNGKey(0),
                info_dict["conv_layers"],
                info_dict["fc_in_sizes"],
            )
            model, state = eqx.tree_deserialise_leaves(f, (model, state))
            return model, state
