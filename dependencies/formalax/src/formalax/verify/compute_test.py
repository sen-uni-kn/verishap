import jax
import jax.numpy as jnp


def simple_relu_network(inputs: jnp.ndarray) -> jnp.ndarray:
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
    print(outputs)
    return outputs


input_lower_bound = jnp.array([-1.0, 1.0])
input_upper_bound = jnp.array([1.0, -1.0])

simple_relu_network(input_lower_bound)
simple_relu_network(input_upper_bound)
