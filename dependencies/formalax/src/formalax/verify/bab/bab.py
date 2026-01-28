#  Copyright (c) 2025. The Formalax Authors.
#  Licensed under the MIT license.
import typing
from collections.abc import Iterator
from typing import Callable, NamedTuple, Protocol

import jax
import jax.numpy as jnp
from jax.extend.core import ClosedJaxpr
from jaxtyping import Array, PyTree, Real

from ...bounds._src._bounds import Bounds
from ...sets import Box
from ...utils.fun_to_jaxpr import fun_to_jaxpr_for_bounding
from .branch_store import ArrayRef, BranchSelection, BranchStore, SimpleBranchStore

_all__ = [
    "BaBCallable",
    "BAB_TARGET",
    "SelectBranches",
    "SplitBranches",
    "ComputeBoundsBaB",
    "MakeForBaB",
    "bab",
]


# ==============================================================================
# Interface Utils
# ==============================================================================


def fun_to_jaxpr_for_bab(
    bab_name: str = "Branch and Bound",
    generator: bool = True,
    with_params_args: bool = False,
):
    """A decorator that converts functions to jaxprs for branch and bound (BaB) functions.

    The function after decoration accepts both jaxprs and functions as argument.

    The decorated function needs to accept a jaxpr as first argument.
    This decorator will check whether a passed first argument is a function
    or a jaxpr.
    If it is a jaxpr, it passes it on directly,
    If it is a function, the function is traced and the resulting jaxpr is given
    to the decorated function.

    Args:
        bab_method: The name of the BaB method that is used.
            This name is included in the docstring of the wrapper and in the wrapper's
            function name (converted to snake case).
        generator: Whether the decorated function is a generator function.
            Default is True.
        with_params_args: Whether the return value of the wrapped BaB function
            has a leading ``params`` argument that should be disregared when converting
            functions to jaxprs.
    """
    # jax_util.wraps docstring argument:
    # {{fun}} placeholder is the function name; {{doc}} is the original docstring.
    docstring = (
        f"Applies {bab_name} to {{fun}}.\n\n\n"
        f"Original documentation of {{fun}}\n"
        f"--------------------------------------------------------------------------------\n"
        f"\n {{doc}}"
    )
    return fun_to_jaxpr_for_bounding(bab_name, docstring, with_params_args, generator)


# ==============================================================================
# Interface
# ==============================================================================


class BaBCallable(Protocol):
    """Type signature of a branch and bound (BaB) callable.

    This is the type signature of the functions created by
    ``bab``.
    Each ``BaBCallable`` is connected to a ``jaxpr``,
    on which it performs branch and bound.
    See ``bab`` for more details.
    """

    def __call__(
        self, *args: Bounds | Real[Array, "..."]
    ) -> Iterator[tuple[Bounds | Real[Array, "..."], ...]]:
        """
        Performs branch and bound (BaB) on the
        ``jaxpr`` encapsulated by this ``BaBCallable`` using the input
        bounds and concrete arguments provided as ``args``.

        Args:
            *args: Input bounds and concrete arguments for which to perform
                backwards bound propagation.
                Each element of ``args`` corresponds to one input variable of ``jaxpr``.

        Returns:
            Returns a generator that yields a sequence of improving bounds on the BaB target.
            The bounds are tuples of ``Bounds``, one for each output variable of ``jaxpr``.
            If an output variable does not depend on bounded inputs, the yielded value
            for that output variable may also be an ``Array`` instead of a ``Bounds`` instance.
        """
        ...


BAB_TARGET = typing.Literal["min", "max", "img"]


class SelectBranches[AR: ArrayRef, BS: BranchSelection](Protocol):
    def __call__(
        self, branches: PyTree[AR], out_lbs: AR, out_ubs: AR, batch_size: int
    ) -> BS: ...


class SplitBranches[BD: PyTree[Array]](Protocol):
    def __call__(self, branches: BD) -> BD: ...


class ComputeBoundsBaB[BD: PyTree[Array]](Protocol):
    """Computes lower and upper bounds for branches.

    The type argument ``BD`` is the type of the branch data.
    """

    def __call__(self, branches: BD) -> Bounds:
        """Computes lower and an upper bounds for ``branches``."""
        ...


class MakeForBaB[T](Protocol):
    """Template for factory functions that create objects for branch and bound."""

    def __call__(
        self,
        jaxpr: ClosedJaxpr,
        *args: Bounds | Real[Array, "..."],
        target: BAB_TARGET,
    ) -> T:
        """Creates an object for branch and bound.

        Args:
            jaxpr: The jaxpr to compute bounds on.
            *args: The input space of the jaxpr for which to compute bounds.
            target: What to bound. The minimum of each ``jaxpr`` output (``"min"``),
                the maximum of each ``jaxpr`` output (``"max"``),
                or the range of outputs of ``jaxpr`` (function image, ``"img"``).
        """
        ...


class _BaBBranchData[BD: PyTree[Array]](NamedTuple):
    data: BD
    out_lbs: Real[Array, "b"]
    out_ubs: Real[Array, "b"]


@fun_to_jaxpr_for_bab(bab_name="Branch and Bound")
def bab[BS: BranchSelection, AR: ArrayRef, BD: PyTree[Array]](
    jaxpr: ClosedJaxpr,
    make_root_branch: MakeForBaB[BD],
    select: MakeForBaB[SelectBranches[AR, BS]],
    split: MakeForBaB[SplitBranches[BD]],
    compute_bounds: MakeForBaB[ComputeBoundsBaB[BD]],
    target: typing.Literal["minimum", "maximum", "image"] | BAB_TARGET = "min",
    batch_size: int = 128,
    jit: bool = True,
    make_branch_store: Callable[[BD], BranchStore[BD, BS, AR]] = SimpleBranchStore,
) -> BaBCallable:
    """Perform branch and bound on ``jaxpr``.

    The ``jaxpr`` must have a single output variable.
    This output variable must be scalar, or a vector that is a batch of scalars.

    Args:
        jaxpr: The jaxpr to compute bounds on. Must have a single output variable.
            This output variable must be scalar, or a vector that is a batch of scalars.
        *args: The input space of the jaxpr for which to compute bounds.
        make_root_branch: Creates the root branch.
            Needs to add a leading batch axis of size 1 to each array in the root branch.
        select: A factory function creating the procedure for selecting branches.
        split: A factory function creating the procedure for selecting splits and splitting branches.
        compute_bounds: A factory function creating the procedure for computing bounds.
        target: What to bound. The minimum of each ``jaxpr`` output (``"minimum"``, ``"min"``),
            the maximum of each ``jaxpr`` output (``"maximum"``, ``"max"``),
            or the range of outputs of ``jaxpr`` (function image, ``"image"``, ``"img"``).
        batch_size: The number of branches to consider at a time.
        jit: Whether to jit the branching step function.
        make_branch_store: Creates a branch store given the root branch.
    """
    match target:
        case "minimum" | "maximum":
            target = target[:3]
        case "image" | "img":
            target = "img"
    if target not in ("min", "max", "img"):
        raise ValueError(f"Unknown target: {target}")

    def bab_fun(*args: Bounds | Real[Array, "..."]):
        return _bab(
            jaxpr,
            *args,
            make_root_branch=make_root_branch,
            make_select=select,
            make_split=split,
            make_compute_bounds=compute_bounds,
            target=target,
            batch_size=batch_size,
            jit=jit,
            make_branch_store=make_branch_store,
        )

    bab_fun.__doc__ = BaBCallable.__doc__
    return bab_fun


# ==============================================================================
# Implementation
# ==============================================================================


def _bab[BS: BranchSelection, AR: ArrayRef, BD: PyTree[Array]](
    jaxpr: ClosedJaxpr,
    *args: Bounds | Real[Array, "..."],
    make_root_branch: MakeForBaB[BD],
    make_select: MakeForBaB[SelectBranches[AR, BS]],
    make_split: MakeForBaB[SplitBranches[BD]],
    make_compute_bounds: MakeForBaB[ComputeBoundsBaB[BD]],
    target: BAB_TARGET = "min",
    batch_size: int = 128,
    jit: bool = True,
    make_branch_store: Callable[[BD], BranchStore[BD, BS, AR]] = SimpleBranchStore,
) -> Iterator[tuple[Bounds | Real[Array, "..."], ...]]:
    """The function returned by ``bab``."""
    if len(jaxpr.out_avals) > 1:
        raise ValueError("bab requires a jaxpr with a single output variable.")

    select = make_select(jaxpr, *args, target=target)
    split = make_split(jaxpr, *args, target=target)
    compute_bounds_fn = make_compute_bounds(jaxpr, *args, target=target)

    def compute_bounds(branches: BD) -> tuple[Real[Array, "b"], Real[Array, "b"]]:
        out_bounds = compute_bounds_fn(branches)
        out_lb, out_ub = out_bounds[0].concrete
        batch_size = out_lb.shape[0]
        assert out_lb.size == batch_size, "Jaxpr of BaB must have a scalar output."
        assert out_ub.size == batch_size, "Jaxpr of BaB must have a scalar output."
        out_lb, out_ub = out_lb.reshape(batch_size), out_ub.reshape(batch_size)
        return out_lb, out_ub

    def prune(
        branches: PyTree[ArrayRef, "..."],
        best_lb: Real[Array, ""],
        best_ub: Real[Array, ""],
    ) -> BS:
        infeasible = branches.out_lbs > branches.out_ubs
        match target:
            case "min":
                return infeasible | (branches.out_lbs > best_ub)
            case "max":
                return infeasible | (branches.out_ubs < best_lb)
            case "img":
                return infeasible
            case _:
                raise AssertionError()

    def bab_step(
        batch: BD, num_branches: int
    ) -> tuple[BD, tuple[Real[Array, "b"], Real[Array, "b"]]]:
        new_branches = split(batch)
        new_bounds = compute_bounds(new_branches)
        return new_branches, new_bounds

    if jit:
        # prune and best_bounds should not be jitted because
        # the shapes of the passed branches change for every call
        bab_step_jit = jax.jit(bab_step)
        bab_step_no_jit = bab_step

        def bab_step(batch: BD, num_branches: int) -> tuple[BD, Bounds]:
            if num_branches < batch_size:
                # avoid recompilation when there are few branches
                return bab_step_no_jit(batch, num_branches=num_branches)
            else:
                return bab_step_jit(batch, num_branches=num_branches)

    def best_bounds(
        branches: _BaBBranchData[BD],
    ) -> tuple[Real[Array, "*axs"], Real[Array, "*axs"]]:
        match target:
            case "min":
                return branches.out_lbs.min(), branches.out_ubs.min()
            case "max":
                return branches.out_lbs.max(), branches.out_ubs.max()
            case "img":
                return branches.out_lbs.min(), branches.out_ubs.max()
        raise AssertionError()

    root_data = make_root_branch(jaxpr, *args, target=target)
    out_lb, out_ub = compute_bounds(root_data)
    root_branch = _BaBBranchData(root_data, out_lb, out_ub)

    branches: BranchStore = make_branch_store(root_branch)
    best_lb, best_ub = out_lb, out_ub
    while True:
        to_prune = prune(branches.data, best_lb, best_ub)
        branches.remove(to_prune)

        if len(branches) == 0:
            if target in ("min", "max") and not jnp.isclose(best_lb, best_ub):
                raise RuntimeError(
                    "Bounds did not converge after all branches were pruned."
                )
            break

        selected = select(*branches.data, batch_size=batch_size)
        batch = branches.pop(selected)

        new_branches, (batch_lb, batch_ub) = bab_step(batch.data, len(selected))
        new_branches = _BaBBranchData(new_branches, batch_lb, batch_ub)
        branches.add(new_branches)

        best_lb, best_ub = best_bounds(branches.data)
        yield Box(best_lb, best_ub)

    while True:
        yield Box(best_lb, best_ub)
