#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
from typing import Any, Callable, TypeGuard

import jax.tree_util

__all__ = ("Zero", "zero", "is_zero", "apply_non_zero")


@jax.tree_util.register_pytree_node_class
class Zero:
    """A stand-in for a scalar zero.

    This class should only be used depending on structure, and not on data.
    For example, if a value is always zero for a certain jaxpr, you can use
    ``Zero``.
    However, if a value is only somtimes zero, depending on the value of another
    variable in the jaxpr, do not use ``Zero``.
    """

    def __eq__(self, other):
        return isinstance(other, Zero)

    def __hash__(self):
        return 0

    def __repr__(self):
        return "Zero"

    def __add__(self, other):
        return other

    def __radd__(self, other):
        return other

    def __bool__(self):
        return False

    def __int__(self):
        return 0

    def __float__(self):
        return 0.0

    # MARK: Jaxpr Compatibility

    def tree_flatten(self):
        return (), ()

    @classmethod
    def tree_unflatten(cls, _, __):
        return cls()


zero = Zero()


def is_zero(x: Any) -> TypeGuard[Zero]:
    """Whether ``x`` is a ``Zero`` instance."""
    return isinstance(x, Zero)


def apply_non_zero[T](
    f: Callable[[T, ...], T], *args, **kwargs
) -> Callable[[T | Zero], T | Zero]:
    return lambda x: f(x, *args, **kwargs) if not is_zero(x) else x
