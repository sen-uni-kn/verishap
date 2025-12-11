# Copyright 2025 David Boetius
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class MLP(eqx.Module):
    """A Multi-Layer Perceptron model for classifying CIFAR-10 images."""

    layers: list
    input_dim: int
    output_dim: int
    hidden_dim: int
    hidden_layers: int

    def __init__(
        self,
        key,
        input_dim=3*32*32,
        output_dim=1,
        hidden_dim=512,
        hidden_layers=3,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers

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
        x = jnp.ravel(x)
        for layer in self.layers:
            x = layer(x)
        return x.squeeze(-1)
