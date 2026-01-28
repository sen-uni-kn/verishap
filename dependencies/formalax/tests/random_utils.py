#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
import warnings
from typing import Generator, Literal

import jax.numpy as jnp
import jax.random
import numpy as np


def get_seed(string):
    """Use a string to obtain a deterministic random seed."""
    # Calculate a simple hash of the string.
    # Don't use Python hash, as isn't consistent across sessions
    # Based on djb2: http://www.cse.yorku.ca/~oz/hash.html
    val = np.int64(5381)
    for char in string:
        # ignore integer overflow warnings
        with warnings.catch_warnings(action="ignore", category=RuntimeWarning):
            val = ((val << 5) + val) + ord(char)
    return val


def rng_key_iter(seed: str | int) -> Generator[int, None, None]:
    if isinstance(seed, str):
        seed = get_seed(seed)
    key = jax.random.key(seed)
    while True:
        key, subkey = jax.random.split(key)
        yield subkey


def rand_uniform(shape, rng_key, lower=-10, upper=10):
    return lower + (upper - lower) * jax.random.uniform(rng_key, shape)


def rand_interval(
    shape,
    rng_key,
    mid_lower=-10,
    mid_upper=10,
    max_range=10,
    sign: Literal["pos", "neg", "any"] = "any",
):
    mid_key, range_key = jax.random.split(rng_key)
    mid = rand_uniform(shape, mid_key, mid_lower, mid_upper)
    range_ = max_range * jax.random.uniform(range_key, shape)
    match sign:
        case "pos":
            range_ = jnp.where(mid - range_ <= 0, 1e-4, range_)
        case "neg":
            range_ = jnp.where(mid + range_ >= 0, 1e-4, range_)
    return mid - range_, mid + range_
