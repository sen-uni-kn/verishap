#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
import jax.numpy as jnp

from formalax import Box
from formalax.bounds.utils import is_bounds


def batch_bounds(bounded_args):
    """Adds a leading dimension to all ``Bounds`` instances."""

    def add_batch_dim(bounds):
        return Box(*(jnp.expand_dims(b, 0) for b in bounds))

    return [add_batch_dim(arg) if is_bounds(arg) else arg for arg in bounded_args]
