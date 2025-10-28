# Copyright 2025 David Boetius
import io

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import ruamel.yaml as yaml


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
