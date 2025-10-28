# Copyright 2025 David Boetius
import equinox as eqx
import jax
from jaxtyping import Array, Float


def zscore_norm(x: Float[Array, " n"], mean: Float[Array, " n"], std: Float[Array, " n"]) -> Float[Array, " n"]:
    return (x - mean) / std

def zscore_unnorm(x: Float[Array, " n"], mean: Float[Array, " n"], std: Float[Array, " n"]) -> Float[Array, " n"]:
    return x * std + mean


class MLP(eqx.Module):
    """A Multi-Layer Perceptron model for classifying tabular data."""

    layers: list

    def __init__(
        self,
        input_dim,
        output_dim,
        key,
        hidden_dim=32,
        hidden_layers=2,
        input_normalizer=None,
        output_normalizer=None,
    ):
        keys = jax.random.split(key, hidden_layers + 1)
        layer_sizes = [input_dim] + [hidden_dim] * hidden_layers + [output_dim]
        layers = []
        if input_normalizer is not None:
            layers.append(input_normalizer)
        for i in range(1, len(layer_sizes)):
            layers.append(
                eqx.nn.Linear(layer_sizes[i - 1], layer_sizes[i], key=keys[i])
            )
            layers.append(jax.nn.relu)
        layers = layers[:-1]  # remove training ReLU layer
        if output_normalizer is not None:
            layers.append(output_normalizer)
        self.layers = layers

    def __call__(self, x: Float[Array, " n"]) -> Float[Array, " m"]:
        for layer in self.layers:
            x = layer(x)
        return x
