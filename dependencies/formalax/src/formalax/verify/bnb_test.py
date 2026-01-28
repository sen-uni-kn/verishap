import jax
import jax.numpy as jnp

from .bab import *


def simple_relu_network(inputs: jnp.ndarray, *args) -> jnp.ndarray:
    """
    A simple two-layer neural network with ReLU activation.
    """
    # Define weights and biases
    W1 = jnp.array([[1.0, -1.0], [0.5, 0.5]])
    b1 = jnp.array([0.0, 0.0])
    W2 = jnp.array([[1.0], [-1.0]])
    b2 = jnp.array([0.0])

    # Compute forward pass
    hidden = jax.nn.relu(jnp.dot(inputs, W1) + b1)
    outputs = jnp.dot(hidden, W2) + b2
    return outputs


input_lower_bound = jnp.array([-1.0, -1.0])  # Lower bounds for inputs
input_upper_bound = jnp.array([1.0, 1.0])  # Upper bounds for inputs

network_bounds = NetworkBounds(
    batch_size=128,
    split_heuristic="longest-edge",
    auto_lirpa_method="CROWN",
    bound_range="minimum",
    device=jax.devices()[0],
)

bounds_generator = network_bounds.bound(
    network=simple_relu_network, input_bounds=(input_lower_bound, input_upper_bound)
)


for step, (lb, ub) in enumerate(bounds_generator):
    print(f"Step {step}:")
    print(f"Lower Bound:\n{lb}")
    print(f"Upper Bound:\n{ub}")

    if step == 5:
        break
