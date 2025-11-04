#  Copyright (c) 2025. David Boetius.
#  Licensed under the MIT license.
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool


def argmax_k(array: Array, k: int, approx: bool = True) -> Bool[Array, " b"]:
    """Returns the mask of the k largest elements in the array."""
    if array.ndim > 2 or (array.ndim == 2 and array.shape[1] != 1):
        raise ValueError(
            f"argmax_k is unsupported for data with shape {array.shape}."
        )

    max_k = jax.lax.approx_max_k if approx else jax.lax.top_k
    indices = max_k(array, k=k)[1]
    indices = indices[:k]  # approx_max_k may return more than k indices

    mask = jnp.zeros(array.shape[0], dtype=bool).at[indices].set(True)
    return mask


def argmin_k(array: Array, k: int, approx: bool = True) -> Bool[Array, " b"]:
    """Returns the mask of the k largest elements in the array."""
    return argmax_k(-array, k, approx)
