#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Real

from .protocols import MinMaxSplittable

__all__ = ("Box",)


@jax.tree_util.register_dataclass
@dataclass(eq=False, frozen=True, slots=True)
class Box[T: Real[Array, "..."]]:
    """A box or hyperrectangle set.

    ``Boxes`` can be split into lower and upper bound using

        >>> box = Box(jnp.array(0.0), jnp.array(1.0))
        >>> lb, ub = box

    Similarly, you can iterate over a ``Box``.
    For example,

        >>> box = Box(jnp.array(0.0), jnp.array(1.0))
        >>> box2 = Box(*[a + 1 for a in box])

    ``Box`` does not perform any checks that ``lower_bound`` is less than
    or equal to ``upper_bound``.

    ``Box`` is the prototypical ``Bounds`` implementation.

    Args:
        lower_bound: The smallest element in all dimensions of the set.
        upper_bound: The largest element in all dimensions of the set.
    """

    # --------------------------------------------------------------------------
    # Bounds Implementation
    # --------------------------------------------------------------------------

    lower_bound: T
    upper_bound: T

    @property
    def concrete(self) -> MinMaxSplittable[T]:
        return self

    # --------------------------------------------------------------------------
    # Array Attributes
    # --------------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return self.lower_bound.shape

    @property
    def dtype(self) -> jnp.dtype:
        return self.lower_bound.dtype

    # --------------------------------------------------------------------------
    # MinMaxSplittable Implementation
    # --------------------------------------------------------------------------

    def __iter__(self):
        yield self.lower_bound
        yield self.upper_bound

    def __getitem__(self, index: int) -> T:
        match index:
            case 0:
                return self.lower_bound
            case 1:
                return self.upper_bound
            case _:
                raise IndexError(f"Index must be 0 or 1, got {index}")

    # --------------------------------------------------------------------------
    # HasProjection Implementation
    # --------------------------------------------------------------------------

    def project(self, x: T) -> T:
        return jnp.clip(x, self.lower_bound, self.upper_bound)

    # --------------------------------------------------------------------------
    # HasRandomSample Implementation
    # --------------------------------------------------------------------------

    def uniform_sample(self, key: jax.Array, n: int = 1) -> T:
        shape = (n,) + self.shape
        # this excludes the upper bound (not great but ok)
        return jax.random.uniform(
            key, shape, self.dtype, minval=self.lower_bound, maxval=self.upper_bound
        )

    # --------------------------------------------------------------------------
    # Comparison
    # --------------------------------------------------------------------------

    def __eq__(self, other):
        return (
            isinstance(other, Box)
            and self.lower_bound == other.lower_bound
            and self.upper_bound == other.upper_bound
        )
