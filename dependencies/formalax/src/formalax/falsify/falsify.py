from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from differential_evolution_pgd_attack import de_pgd_counterexample
from fgsm import fgsm_counterexample
from pgd_attack import pgd_counterexample
from utils import Bounds


def nn(x: jnp.ndarray) -> jnp.ndarray:
    w = jnp.array([[1.0, -1.0]])
    b = jnp.array([-0.6])
    return jax.nn.relu(x @ w.T) + b


def nonneg_satisfaction(
    inputs: jnp.ndarray, network: Callable
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    outs = network(inputs)
    vals = jnp.min(outs, axis=1)
    sat_mask = vals >= 0
    return vals, sat_mask


bounds = Bounds(lower_bound=jnp.array([-1.0, -1.0]), upper_bound=jnp.array([1.0, 1.0]))


cx, status = fgsm_counterexample(nn, bounds, nonneg_satisfaction)

print("Status:", status)
print("Counterexamples:\n", cx)


cx, status = pgd_counterexample(
    nn,
    bounds,
    nonneg_satisfaction,
    steps=30,
    num_restarts=20,
    optimizer="adam",
)

print("Status:", status)
print("Counterexamples:\n", cx)


cx, status = de_pgd_counterexample(
    nn,
    bounds,
    nonneg_satisfaction,
    optimizer="adam",
    population_size=20,
    iterations=15,
    local_steps=20,
    progress_bar=True,
    single_counterexample=False,
)

print("Status:", status)
print("Counterexamples:\n", cx)
