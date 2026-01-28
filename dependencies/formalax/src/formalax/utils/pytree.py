#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
from typing import Any, Callable

import jax
from jaxtyping import PyTreeDef

from .future_store import FutureStore


def flatten_keep_none(tree: Any) -> tuple[list, jax.tree_util.PyTreeDef]:
    """Flattens a pytree while keeping ``None`` leaves."""
    return jax.tree.flatten(tree, is_leaf=lambda x: x is None)


def flatten_fun(
    fun: Callable, in_tree: PyTreeDef | None, is_leaf: Callable | None = None
) -> tuple[Callable, PyTreeDef]:
    """Wrap a function so that it accepts flattened arguments and returns flattened output.

    Args:
        fun: The function to wrap.
        in_tree: The tree of the input arguments.
        is_leaf: The ``is_leaf`` argument for flattening the output of ``fun``.

    Returns:
        A the wrapped function and a thunk that returns the pytree of the
        output of ``fun``.
        The thunk can only be called after the wrapped function has been called.
    """
    out_tree_thunk = FutureStore()

    def flattened(*args, **kwargs):
        if in_tree is not None:
            args = jax.tree.unflatten(in_tree, args)
        ans = fun(*args, **kwargs)
        out_flat, out_tree = jax.tree.flatten(ans, is_leaf)
        out_tree_thunk.assign(out_tree)
        return out_flat

    return flattened, out_tree_thunk
