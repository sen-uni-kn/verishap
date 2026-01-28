#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
from collections.abc import Iterable
from typing import Any, Protocol, TypeGuard

from jaxtyping import PRNGKeyArray

__all__ = (
    "HasProjection",
    "has_projection",
    "MinMaxSplittable",
    "CanDrawRandomSample",
    "can_draw_random_sample",
)


class HasProjection[T](Protocol):
    """A type that implements a projection method."""

    def project(self, x: T) -> T:
        """Projects ``x`` onto the set.

        The return value of this method is guaranteed to be in the set ``self``.

        Args:
            x: The value to project.

        Returns:
            The projected value.
        """
        ...


def has_projection(x: Any) -> TypeGuard[HasProjection]:
    """Checks whether ``x`` implements ``HasProjection``."""
    return hasattr(x, "project")


class MinMaxSplittable[T](Protocol):
    """A set type that can be split into its minimal and maximal elements.

    Set implementing ``MinMaxSplittable`` can be split into the minimal element of
    the set and the maximal element of the set using

        >>> x = ...  # some type implementing MinMaxSplittable
        >>> lb, ub = x

    Similarly, you can iterate over a ``MinMaxSplittable``.
    For example,

        >>> box = ...  # some type implementing MinMaxSplittable
        >>> box2 = [a + 1 for a in box]
    """

    def __iter__(self) -> Iterable[T]:
        """Needs to yield exactly two elements."""
        ...

    def __getitem__(self, index: int) -> T:
        """Returns the element at the given index (``0`` or ``1``)."""
        ...


class CanDrawRandomSample[T](Protocol):
    """A set type that allows drawing random samples from it."""

    def uniform_sample(self, key: PRNGKeyArray, n: int = 1) -> T:
        """Draw a random sample from this set.

        The samples are drawn from a uniform distribution over this set.

        Args:
            key: A jax pseudo random number generator key.
            n: The number of samples to draw.
        Returns:
            An array of samples with a leading batch dimension of size `n`.
        """


def can_draw_random_sample(x: Any) -> TypeGuard[CanDrawRandomSample]:
    """Checks whether ``x`` implements ``CanDrawRandomSample``."""
    return hasattr(x, "uniform_sample")
