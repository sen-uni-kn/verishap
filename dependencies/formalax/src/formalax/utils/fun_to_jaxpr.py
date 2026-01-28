#  Copyright (c) 2025. The Formalax Authors.
#  Licensed under the MIT license.
import functools
import re
from typing import Any, Callable

import jax
from jax._src import util as jax_util
from jax.extend.core import ClosedJaxpr
from jaxtyping import Array, PyTree, PyTreeDef, Real

from ..bounds._src._bounds import Bounds, example_args, is_bounds


def make_jaxpr_with_bounds(
    fun: Callable,
    fun_args: PyTree[Bounds | Real[Array, "..."]],
    fun_kwargs: dict[str, Any],
) -> tuple[ClosedJaxpr, list[Bounds | Real[Array, "..."]], PyTreeDef]:
    """Creates a jaxpr for a function using ``Bounds`` arguments.

    Returns:
        The jaxpr, flattened ``fun_args``, and the pytree of ``fun``'s output.
    """
    args_flat, in_tree = jax.tree.flatten(fun_args, is_leaf=is_bounds)

    xs = example_args(args_flat)
    xs = jax.tree.unflatten(in_tree, xs)
    jaxpr, out_shapes = jax.make_jaxpr(fun, return_shape=True)(*xs, **fun_kwargs)
    _, out_tree = jax.tree.flatten(out_shapes)

    return jaxpr, args_flat, out_tree


def bounding_wrapper_of(fun: Callable, docstring: str, fun_name: str):
    """Function decorator for wrappers that compute bounds on ``fun``.

    Args:
        fun: The original function whose bounds are to be computed.
        docstring: The string to prepend to the original docstring.
            Can include the placeholders {{fun}} and {{doc}} for the function name and
            original docstring, respectively.
    """
    # Thanks to https://stackoverflow.com/a/3305731/10550998 for
    # converting strings to python identifies.
    fun_name = re.sub(r"\W+|^(?=\d)", "_", fun_name.lower())
    fun_name = f"{fun_name}_{{fun}}"

    def decorator(wrapper_fun: Callable):
        # jax_util.wraps docstring argument:
        # {{fun}} placeholder is the function name; {{doc}} is the original docstring.
        return jax_util.wraps(fun, doc=docstring, namestr=fun_name)(wrapper_fun)

    return decorator


def fun_to_jaxpr_for_bounding(
    bounding_name: str,
    docstring: str,
    with_params_args: bool = False,
    generator: bool = False,
):
    """A decorator that converts functions to jaxprs for functions that compute bounds.

    After decroation, the decorated function accepts both jaxprs and functions
    as argument.

    The decorated function needs to accept a jaxpr as first argument.
    This decorator will check whether a passed first argument is a function
    or a jaxpr.
    If it is a jaxpr, it passes it on directly,
    If it is a function, the function is traced and the resulting jaxpr is given
    to the decorated function.

    Args:
        bounding_name: The name of the bounding procedure that this decorator decorates.
            For example ``"backwards LiRPA"`` or ``CROWN-IBP``.
        docstring: The string to prepend to the original docstring.
            Can include the placeholders {{fun}} and {{doc}} for the function name and
            original docstring, respectively.
        with_params_args: Whether the return value of the wrapped bounding function
            has a leading ``params`` argument that should be disregarded when converting
            functions to jaxprs.
        generator: Whether the decorated function is a generator function.
            If True, the decorated function will yield from the decorated function.
    """

    # the decorated function is also a higher order function: it returns a function
    # that computes bounds.

    def decorator(bounds_fun: Callable[[ClosedJaxpr, ...], Callable]):
        @functools.wraps(bounds_fun)
        def bounds_fun_wrapper(fun_or_jaxpr, *bounds_fun_args, **bounds_fun_kwargs):
            if isinstance(fun_or_jaxpr, ClosedJaxpr):
                # returns a callable (+ potentially other values)
                return bounds_fun(fun_or_jaxpr, *bounds_fun_args, **bounds_fun_kwargs)

            fun = fun_or_jaxpr

            @bounding_wrapper_of(fun, docstring, bounding_name)
            def compute_bounds_wrapper(*fun_args, **fun_kwargs):
                params = None
                if with_params_args:
                    params, *fun_args = fun_args

                jaxpr, args_flat, out_tree = make_jaxpr_with_bounds(
                    fun, fun_args, fun_kwargs
                )
                compute_bounds = bounds_fun(
                    jaxpr, *bounds_fun_args, **bounds_fun_kwargs
                )

                if with_params_args:
                    out_bounds_flat = compute_bounds(params, *args_flat)
                else:
                    out_bounds_flat = compute_bounds(*args_flat)

                def as_tuple(val: tuple | Any) -> tuple:
                    if not isinstance(val, tuple):
                        return (val,)
                    return val

                if not generator:
                    return jax.tree.unflatten(out_tree, as_tuple(out_bounds_flat))
                else:
                    return (
                        jax.tree.unflatten(out_tree, as_tuple(flat_bounds))
                        for flat_bounds in out_bounds_flat
                    )

            return compute_bounds_wrapper

        return bounds_fun_wrapper

    return decorator
