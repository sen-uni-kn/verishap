#  Copyright (c) 2025. The Formalax Authors.
#  Licensed under the MIT license.
import jax
import jax.numpy as jnp


def expand_as_diagonal(
    x: jax.Array, axis: int, new_axis1: int = -2, new_axis2: int = -1
) -> jax.Array:
    """Expands an axis of ``x`` into the diagonal of two new axes.

    If ``x`` has shape ``(a, b, c)`` and ``axis = 1``, this function returns
    an array ``y`` with shape ``(a, c, b, b)`` where

        y[:, :, i, i] = x[:, i, :]

        y[:, :, i, j] = 0 for i != j.

    The position of the new axes can be specified using ``new_axis1`` and
    ``new_axis2``.

    Args:
        x: The array to expand.
        axis: The axis to expand.
        new_axis1: The index of the first new axis in the shape of the output.
        new_axis2: The index of the second new axis in the shape of the output.

    Returns:
        The array ``x`` with the axis ``axis`` expanded into the diagonal
        of the new axes.
    """
    if new_axis1 < 0:
        new_axis1 = len(x.shape) + new_axis1
    if new_axis2 < 0:
        new_axis2 = len(x.shape) + new_axis2
    n = x.shape[axis]

    x = jnp.moveaxis(x, axis, -1)
    x_broadcast = jnp.reshape(x, (*x.shape, 1))
    eye = jnp.eye(n, dtype=x.dtype)
    out = x_broadcast * eye

    out_axes = list(range(len(x.shape) - 1))
    out_axes.insert(new_axis1, len(out.shape) - 2)
    out_axes.insert(new_axis2, len(out.shape) - 1)
    return jnp.permute_dims(out, tuple(out_axes))
