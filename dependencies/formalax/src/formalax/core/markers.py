#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
from functools import partial
from typing import Callable

import jax
import jax.extend.core
import jax.extend.linear_util as lu
from jax.extend.core import primitives

__all__ = (
    "Marker",
    "relu_marker",
    "detect_jax_nn_relu_call",
    "detect_jax_nn_relu_eqn",
    "markup_primitive",
)


class Marker:
    """Marker for predefined computations.

    Markers are used to indicate a certain computation that could be
    decomposed into further Jax primitives but should be handled as is.

    The typical example is ReLU, which is not a JAX primitive but is treated
    as a primitive operation in many bound propagation algorithms.
    """

    __slots__ = ("name", "multiple_results")

    def __init__(self, name, multiple_results=False):
        self.name = name
        self.multiple_results = multiple_results

    def __str__(self):
        return self.name

    def __repr__(self):
        if self.multiple_results:
            return f"Marker({self.name}, multiple_results=True)"
        else:
            return f"Marker({self.name})"


relu_marker = Marker("relu")


def detect_jax_nn_relu_eqn(eqn: jax.extend.core.JaxprEqn) -> bool:
    """Checks whether ``eqn`` is a ``jax.nn.relu`` call."""
    return (
        eqn.primitive == jax.custom_derivatives.custom_jvp_call_p
        and len(sub_eqns := eqn.params["call_jaxpr"].jaxpr.eqns) == 1
        and (sub_eqn := sub_eqns[0]).primitive == primitives.jit_p
        and "name" in (sub_params := sub_eqn.params)
        and sub_params["name"] == "relu"
    )


def detect_jax_nn_relu_call(f: Callable):
    """Checks whether ``f`` is ``jax.nn.relu`` or a ``jax.nn.relu`` wrapper."""
    return (f == jax.nn.relu.fun) or (
        isinstance(f, partial)
        and len(f.args) == 1
        and isinstance((jaxpr := f.args[0]), jax.extend.core.ClosedJaxpr)
        and len((eqns := jaxpr.eqns)) == 1
        and "name" in (params := eqns[0].params)
        and params["name"] == "relu"
    )


def markup_primitive(
    eqn: jax.extend.core.JaxprEqn,
) -> jax.extend.core.Primitive | Marker:
    """Marks up special operations, such as ``jax.nn.relu`` and otherwise returns
    the primitive of ``eqn``.
    """
    if detect_jax_nn_relu_eqn(eqn):
        return relu_marker
    else:
        return eqn.primitive


def markup_primitive_call(
    primitive: jax.extend.core.Primitive,
    *args,
    **params,
) -> jax.extend.core.Primitive | Marker:
    """Marks up special operations, such as ``jax.nn.relu`` and otherwise returns
    the called primitive.
    """
    if primitive == jax.extend.core.primitives.custom_jvp_call_p:
        fun = args[0]
        if isinstance(fun, lu.WrappedFun):
            fun = fun.f
        if detect_jax_nn_relu_call(fun):
            return relu_marker
    else:
        return primitive
