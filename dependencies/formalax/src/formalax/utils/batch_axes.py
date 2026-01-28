#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
from collections.abc import Container
from types import EllipsisType
from typing import Sequence


def canonicalize_batch_axes(
    batch_axes: int | None | Sequence[int | None | tuple[int, ...]] | EllipsisType,
    reference: Sequence[bool],
) -> tuple[tuple[int, ...], ...] | EllipsisType:
    """Transfers a batch axes argument into a canonical form.

    This is a helper function for functions that accept batch axes as arguments.

    Args:
        batch_axes: The batch axes argument to canonicalize.
            ``None`` means no batch axes and single integers mean a single batch axis.
        reference: A sequence of booleans representing the arrays that ``batch_axes``
            refers to.
            When ``batch_axes`` is an integer ``i``, the boolean value of an element
            determines whether this element has ``(i,)`` (``True``) or ``()`` (``False``)
            as batch axis in the canonical form.
    """
    if batch_axes is ...:
        return ...
    elif batch_axes is None:
        batch_axes = ((),) * len(reference)
    elif isinstance(batch_axes, int):
        batch_axes = tuple((batch_axes,) if has_ba else () for has_ba in reference)
    return tuple(
        () if ba is None else (ba,) if isinstance(ba, int) else ba for ba in batch_axes
    )


def batch_shape(shape: tuple[int, ...], batch_axes: Sequence[int]) -> tuple[int, ...]:
    """Extracts the batch axes from a shape.

    Args:
        shape: The full shape including batch axes.
        batch_axes: The positions of the batch axes.

    Returns:
        The shape of the batch axes.
    """
    return tuple(shape[i] for i in batch_axes)


def non_batch_shape(
    shape: tuple[int, ...], batch_axes: Sequence[int]
) -> tuple[int, ...]:
    """Extracts the non-batch axes from a shape.

    Args:
        shape: The full shape including batch axes.
        batch_axes: The positions of the batch axes.

    Returns:
        The shape without the batch axes.
    """
    return tuple(size for i, size in enumerate(shape) if i not in batch_axes)


def split_shape(
    full_shape: tuple[int, ...],
    batch_axes: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Splits a shape into the batch and non-batch shapes.

    Args:
        full_shape: The full shape including batch axes.
        batch_axes: The positions of the batch axes.

    Returns:
        The batch shape and the non-batch shape.
    """
    return batch_shape(full_shape, batch_axes), non_batch_shape(full_shape, batch_axes)


def full_shape(
    batch_shape: tuple[int, ...],
    non_batch_shape: tuple[int, ...],
    batch_axes: Container[int],
) -> tuple[int, ...]:
    """Combines the batch shape and non-batch shape into the full shape.

    Args:
        batch_shape: The shape of the batch axes.
        non_batch_shape: The shape of the non-batch axes.
        batch_axes: The positions of the batch axes in the full shape.

    Returns:
        The full shape, combining the batch shape and non-batch shape.
    """
    batch_iter = iter(batch_shape)
    non_batch_iter = iter(non_batch_shape)
    return tuple(
        next(batch_iter) if i in batch_axes else next(non_batch_iter)
        for i in range(len(batch_axes) + len(non_batch_shape))
    )
