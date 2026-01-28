#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
import itertools as it
from typing import Any, Protocol, Sequence, TypeGuard

import jax
from jaxtyping import Array, Real, Shaped

from ...sets.protocols import MinMaxSplittable
from ...sets.singleton import Singleton

__all__ = (
    "Bounds",
    "is_bounds",
    "all_as_bounds",
    "example_args",
    "duplicate_for_bounds",
)


class Bounds[T: Real[Array, "..."]](Protocol):
    """A pair of a lower and an upper bound on a set of ``Arrays``.

    The term "concrete bounds" refers to constant bounds in contrast to, for
    example, linear bounds.
    A ``Bounds`` instance does not need to store concrete bounds but may compute
    it from other forms of bounds.
    Consider caching computed concrete bounds.

    ``Bounds`` classes should be registered as PyTree nodes in JAX.
    """

    @property
    def lower_bound(self) -> T:
        """The concrete lower bound, the first element of ``self.concrete``."""
        ...

    @property
    def upper_bound(self) -> T:
        """The concrete upper bound, the second element of ``self.concrete``."""
        ...

    @property
    def concrete(self) -> MinMaxSplittable[T]:
        """The concrete bounds underlying this ``Bounds`` instance."""
        ...


def is_bounds(x: Any) -> TypeGuard[Bounds]:
    """Whether ``x`` provides the properties of ``Bounds``."""
    return (
        hasattr(x, "lower_bound")
        and hasattr(x, "upper_bound")
        and hasattr(x, "concrete")
    )


# ======================================================================================
# Utils
# ======================================================================================


def all_as_bounds[T](*xs: T | Bounds[T]) -> tuple[Bounds[T], ...]:
    """Convert all arguments ``x`` which are not ``Bounds`` to ``Box(x, x)``.

    Args:
        *xs: The arguments to convert. Arguments that are already ``Bounds``
            are left as they are.

    Returns:
        A tuple of ``Bounds`` instances.
    """
    return tuple(x if is_bounds(x) else Singleton(x) for x in xs)


def example_args[T: Shaped](args_flat: Sequence[T | Bounds[T]]) -> list[T]:
    """Replace ``Bounds`` by the lower bound to have array-only arguments."""
    assert not is_bounds(args_flat)
    return [a.lower_bound if is_bounds(a) else a for a in args_flat]


def duplicate_for_bounds[T, U](
    base_values: Sequence[U], with_bounds: Sequence[T | Bounds[T]]
) -> tuple[U, ...]:
    """Duplicates values in ``base_values`` whenever ``with_bounds`` contains a
    ``Bounds`` instance.

    The ``base_values`` and ``with_bounds`` arguments must have the same length.
    """
    assert len(base_values) == len(with_bounds)
    base_values, with_bounds = list(base_values), list(with_bounds)
    duplicates = jax.tree.map(
        lambda x, b: (x, x) if is_bounds(b) else (x,), base_values, with_bounds,
        is_leaf=lambda x: x is None
    )
    return tuple(it.chain.from_iterable(duplicates))
