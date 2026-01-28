import jax
import jax.numpy as jnp

from ._backwards_lirpa import crown_ibp
from .core import Bounds


def simple_relu_network(inputs: jnp.ndarray) -> jnp.ndarray:
    # Define weights and biases
    W1 = jnp.array([[1.0, -1.0], [0.5, 0.5]])
    b1 = jnp.array([0.0, 0.0])
    W2 = jnp.array([[1.0], [-1.0]])
    b2 = jnp.array([0.0])

    # Compute forward pass
    hidden = jax.nn.relu(jnp.dot(inputs, W1) + b1)
    outputs = jnp.dot(hidden, W2) + b2
    return outputs


print(
    crown_ibp(simple_relu_network)(
        Bounds(jnp.array([-1.0, -1.0]), jnp.array([1.0, 1.0]))
    )
)
