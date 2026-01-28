#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Real

from .protocols import MinMaxSplittable

__all__ = ("Singleton",)


@jax.tree_util.register_dataclass
@dataclass(eq=False, frozen=True, slots=True)
class Singleton[T: Real[Array, "..."]]:
    val: T

    # --------------------------------------------------------------------------
    # Array Attributes
    # --------------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return self.val.shape

    @property
    def dtype(self) -> jnp.dtype:
        return self.val.dtype

    # --------------------------------------------------------------------------
    # MinMaxSplittable Implementation
    # --------------------------------------------------------------------------

    def __iter__(self):
        yield self.val
        yield self.val

    def __getitem__(self, index: int) -> T:
        if index not in (0, 1):
            raise IndexError(f"Index must be 0 or 1, got {index}")
        return self.val

    # --------------------------------------------------------------------------
    # Bounds Implementation
    # --------------------------------------------------------------------------

    @property
    def lower_bound(self) -> T:
        return self.val

    @property
    def upper_bound(self) -> T:
        return self.val

    @property
    def concrete(self) -> MinMaxSplittable[T]:
        return self

    # --------------------------------------------------------------------------
    # HasProjection Implementation
    # --------------------------------------------------------------------------

    def project(self, x: T) -> T:
        return self.val

    # --------------------------------------------------------------------------
    # HasRandomSample Implementation
    # --------------------------------------------------------------------------

    def uniform_sample(self, key: jax.Array, n: int = 1) -> T:
        return jnp.repeat(
            jnp.expand_dims(self.val, 0), n, axis=0, total_repeat_length=n
        )

    # --------------------------------------------------------------------------
    # Comparison
    # --------------------------------------------------------------------------

    def __eq__(self, other):
        return isinstance(other, Singleton) and self.val == other.val
