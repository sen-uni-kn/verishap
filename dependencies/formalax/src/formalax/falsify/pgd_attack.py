from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import optax
from utils import Bounds, SatisfactionFunction


def pgd_counterexample(
    network: Callable[[jnp.ndarray], jnp.ndarray],
    input_bounds: Bounds,
    satisfaction_fn: SatisfactionFunction,
    *,
    steps: int = 10,
    num_restarts: int = 10,
    optimizer: str = "sgd",
    progress_bar: bool = True,
    single_counterexample: bool = False,
    input_constraint_projection: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
) -> Tuple[jnp.ndarray, str]:
    lower, upper = input_bounds.lower_bound, input_bounds.upper_bound
    max_radius = jnp.max(upper - lower)
    lr = (max_radius / steps) * 2.0

    if optimizer.lower() == "adam":
        opt = optax.adam(lr)
    elif optimizer.lower() == "rmsprop":
        opt = optax.rmsprop(lr)
    else:
        opt = optax.sgd(lr)

    rng = jax.random.PRNGKey(0)
    cxs = []

    iterator = range(num_restarts)
    if progress_bar:
        from tqdm import tqdm

        iterator = tqdm(iterator)

    for _ in iterator:
        rng, subkey = jax.random.split(rng)
        x = lower + (upper - lower) * jax.random.uniform(subkey, shape=lower.shape)
        x = input_constraint_projection(x[jnp.newaxis, :])
        params = jnp.squeeze(x, axis=0)
        opt_state = opt.init(params)

        def loss_fn(p):
            v, _ = satisfaction_fn(p[jnp.newaxis, :], network)
            return jnp.min(v) if v.ndim > 0 else v

        grad_fn = jax.grad(loss_fn)

        for _ in range(steps):
            grads = grad_fn(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            params = jnp.clip(params, lower, upper)
            params = jnp.squeeze(
                input_constraint_projection(params[jnp.newaxis, :]), axis=0
            )

        sat_val, is_sat = satisfaction_fn(params[jnp.newaxis, :], network)
        if not bool(is_sat):
            cxs.append((params, sat_val))

    if len(cxs) == 0:
        return jnp.array([]), "Unknown"

    cxs.sort(key=lambda t: float(t[1].item()))
    if single_counterexample:
        cxs = [cxs[0]]

    return jnp.stack([c[0] for c in cxs]), "Violation"
