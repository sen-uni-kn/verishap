#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
from typing import Protocol

import jax
from jaxtyping import Array, PyTree

from ._src._bounds import Bounds


class ComputeBounds[B: Bounds](Protocol):
    """Computes bounds on a function or jaxpr.

    Args:
        *args: Bounded arguments for the target function or jaxpr.
            Some arguments may be be fixed.
            These are ``Array``s instead of ``Bounds`` instances.
    Returns:
        Bounds (or fixed values) for each output of the target function or jaxpr.
    """

    def __call__(self, *args: PyTree[Bounds | Array]) -> PyTree[B | Array]: ...


class MakeComputeBounds[B: Bounds](Protocol):
    def __call__(self, jaxpr: jax.core.ClosedJaxpr) -> ComputeBounds[B]: ...
