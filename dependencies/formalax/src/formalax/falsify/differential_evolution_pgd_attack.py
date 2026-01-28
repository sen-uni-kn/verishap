from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from utils import Bounds, SatisfactionFunction


def _get_optax_optimizer(name: str, lr: float):
    name = name.lower()
    if name == "adam":
        return optax.adam(lr)
    if name == "rmsprop":
        return optax.rmsprop(lr)
    return optax.sgd(lr)


def _project_population(
    pop: jnp.ndarray,
    lower: jnp.ndarray,
    upper: jnp.ndarray,
    projection_fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]],
):
    """
    Apply input_constraint_projection clip to bounds.
    This matches the behaviour used elsewhere (projection then clipping).
    """
    if projection_fn is not None:
        pop = projection_fn(pop)
    pop = jnp.clip(pop, lower, upper)
    return pop


def de_pgd_counterexample(
    network: Callable[[jnp.ndarray], jnp.ndarray],
    input_bounds: Bounds,
    satisfaction_fn: SatisfactionFunction,
    *,
    optimizer: str = "sgd",
    population_size: int = 10,
    iterations: int = 10,
    local_steps: int = 10,
    cr: float = 0.9,
    dw: float = 0.8,
    progress_bar: bool = False,
    single_counterexample: bool = False,
    input_constraint_projection: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = (
        lambda x: x
    ),
    rng_key=None,
) -> Tuple[jnp.ndarray, str]:
    """
    Differential Evolution + PGD search for counterexamples.

    Returns (results_array, status_str), where results_array has shape (n_found, dim).
    Status is "Violation" if any violating inputs were discovered, otherwise "Unknown".
    """
    assert population_size >= 4, (
        "population_size must be >= 4 (DE needs 3 distinct other members)"
    )
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    lower = jnp.asarray(input_bounds.lower_bound)
    upper = jnp.asarray(input_bounds.upper_bound)
    lower = jnp.reshape(lower, (-1,))
    upper = jnp.reshape(upper, (-1,))
    dim = lower.shape[0]

    # learning rate scaled with bounds and local_steps
    max_radius = float(jnp.max(upper - lower))
    lr = (max_radius / max(1, local_steps)) * 2.0
    opt = _get_optax_optimizer(optimizer, lr)

    # initialize population
    rng_key, sub = jax.random.split(rng_key)
    unif = jax.random.uniform(sub, shape=(population_size, dim))
    population = lower + (upper - lower) * unif
    if input_constraint_projection is not None:
        population = input_constraint_projection(population)
    population = jnp.clip(population, lower, upper)

    opt_init_v = jax.vmap(lambda p: opt.init(p))
    opt_update_v = jax.vmap(lambda g, s, p: opt.update(g, s, p))
    apply_updates_v = jax.vmap(optax.apply_updates)

    opt_states = opt_init_v(population)

    # loss
    def loss_single(p):
        v, _ = satisfaction_fn(jnp.expand_dims(p, 0), network)
        return jnp.min(v) if jnp.ndim(v) > 0 else v

    grad_single = jax.grad(loss_single)
    grad_v = jax.vmap(grad_single)  # maps (D,) -> (D,) across population

    def scalar_violation_for_population(pop_batch: jnp.ndarray):
        # Returns scalar violation per row
        v, _ = satisfaction_fn(pop_batch, network)
        if jnp.ndim(v) > 1:
            return jnp.min(v, axis=1)
        else:
            return jnp.ravel(v)

    iterator = range(iterations)
    if progress_bar:
        from tqdm import tqdm

        iterator = tqdm(iterator)

    for _ in iterator:
        # Local PGD optimizatio
        for _ in range(local_steps):
            grads = grad_v(population)
            updates, opt_states = opt_update_v(grads, opt_states, population)
            population = apply_updates_v(population, updates)
            population = _project_population(
                population, lower, upper, input_constraint_projection
            )

        current_vals = scalar_violation_for_population(population)
        # DE recombination
        rng_key, perm_keys, r_keys, cross_keys = jax.random.split(rng_key, 4)
        perm_keys = jax.random.split(perm_keys, population_size)
        r_keys = jax.random.split(r_keys, population_size)
        cross_keys = jax.random.split(cross_keys, population_size)

        def sample_triplet(key, i):
            perm = jax.random.permutation(key, population_size)
            is_i = (perm == i).astype(jnp.int32)
            perm = perm[jnp.argsort(is_i)]
            return perm[:3]

        idxs_mat = jax.vmap(sample_triplet)(perm_keys, jnp.arange(population_size))

        a = population[idxs_mat[:, 0]]
        b = population[idxs_mat[:, 1]]
        c = population[idxs_mat[:, 2]]

        def make_mask(rk, ck):
            r = jax.random.randint(rk, (), 0, dim)
            cross_rand = jax.random.uniform(ck, shape=(dim,))
            mask = (jnp.arange(dim) == r) | (cross_rand < cr)
            return mask

        masks = jax.vmap(make_mask)(r_keys, cross_keys)
        candidates = jnp.where(masks, a + dw * (b - c), population)
        candidates = jnp.clip(candidates, lower, upper)
        if input_constraint_projection is not None:
            candidates = input_constraint_projection(candidates)

        candidate_vals = scalar_violation_for_population(candidates)
        replace_mask = candidate_vals <= current_vals

        # Update population where candidate is better
        population = jnp.where(replace_mask[:, None], candidates, population)

    # Final evaluation
    v_batch, s_batch = satisfaction_fn(population, network)
    if jnp.ndim(v_batch) > 1:
        v_flat = jnp.min(v_batch, axis=1)
    else:
        v_flat = jnp.ravel(v_batch)

    s_bool = jnp.asarray(s_batch)
    s_bool = jnp.squeeze(s_bool).astype(bool)

    violated_idx = jnp.where(~s_bool)[0]
    if violated_idx.size == 0:
        return jnp.array([]), "Unknown"

    violated_vals = v_flat[violated_idx]
    order = jnp.argsort(violated_vals)
    ordered_idx = violated_idx[order]
    results = population[ordered_idx]

    if single_counterexample:
        results = results[:1]

    return results, "Violation"
