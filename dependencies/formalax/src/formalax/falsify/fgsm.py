from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from utils import SatisfactionFunction


def fgsm_counterexample(
    network: Callable[[jnp.ndarray], jnp.ndarray],
    input_bounds,
    satisfaction_fn: SatisfactionFunction,
) -> Tuple[jnp.ndarray, str]:
    """
    Generates counterexamples/adversarial examples by walking into
    the direction of the sign of the gradients until hitting the bounds.
    """

    lower = jnp.array(input_bounds.lower_bound)
    upper = jnp.array(input_bounds.upper_bound)
    x = lower + (upper - lower) / 2.0

    grad_fn = jax.grad(
        lambda x: jnp.min(satisfaction_fn(x[jnp.newaxis, :], network)[0])
    )
    grad_x = grad_fn(x)

    cx_candidate = jnp.where(-jnp.sign(grad_x) < 0, lower, upper)
    cx_candidate = jnp.where(-jnp.sign(grad_x) == 0, x, cx_candidate)

    _, is_sat = satisfaction_fn(cx_candidate[jnp.newaxis, :], network)
    if not bool(is_sat[0]):
        return cx_candidate, "Violation"
    else:
        return jnp.array([]), "Unknown"
