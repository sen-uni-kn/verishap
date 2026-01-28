#  Copyright (c) 2025. The Formalax Authors.
#  Licensed under the MIT license.
import functools
import itertools as it
import math
import typing
from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.extend.core
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Int, Real

from ...bounds._src._bounds import Bounds, is_bounds
from ...bounds._src._bwcrown import crown, crown_ibp
from ...bounds._src._ibp import ibp_jaxpr
from ...sets.box import Box
from ...utils.zip import strict_zip
from .bab import (
    BAB_TARGET,
    ArrayRef,
    BaBCallable,
    BranchSelection,
    BranchStore,
    ComputeBoundsBaB,
    MakeForBaB,
    SelectBranches,
    SimpleBranchStore,
    SplitBranches,
    bab,
    fun_to_jaxpr_for_bab,
)
from .select import select_worst

__all__ = [
    "InputSplit",
    "ComputeBoundsForInputSplitting",
    "split_longest_edge",
    "input_splitting_bab",
    "longest_edge",
    "ibp_input_splitting",
    "crown_input_splitting",
    "crown_ibp_input_splitting",
]

# ==============================================================================
# Branch Data
# ==============================================================================


@jax.tree_util.register_dataclass
@dataclass(eq=False, frozen=True, slots=True)
class InputSplit:
    """A branching that partitions the input space into disjoint boxes.

    The arrays stored in this class are generally flattened.
    Due to being stored in a branch store, this means that the arrays have
    two axes: a learning batch/branches axis and a data axis.
    The only exception to this is the root branch, which is created without
    a leading branch axis.

    If the jaxpr that branch and bound is applied to has multiple inputs,
    the flattened input arrays are concatenated.

    Args:
        in_lbs: The flattened and concatenated lower bounds of the input space.
        in_ubs: The flattened and concatenated upper bounds of the input space.
        fixed_args: Arguments of ``jaxpr`` that do not have bounds.
            These are not flattened.
            This is static data that does not change during branch and bound.
        bounds_shapes: The shapes of the unflattened bounded inputs.
        has_bounds: Whether each argument of ``jaxpr`` is a bounds instance.
    """

    in_lbs: Real[Array, "b d"]
    in_ubs: Real[Array, "b d"]
    fixed_args: tuple[Array, ...] = field(
        metadata={"static": True}
    )  # for jax.tree_util
    bounds_shapes: tuple[tuple[int, ...], ...] = field(metadata={"static": True})
    has_bounds: tuple[bool, ...] = field(metadata={"static": True})

    def as_input(self) -> tuple[Box[Array] | Array, ...]:
        """Reformats the data in this instance as arguments of the underlying jaxpr.

        Returns:
            A tuple of ``Bounds`` and ``Array``s in the correct order.
            All bounds are reshaped to their original shapes.
        """
        num_elems = [math.prod(shape) for shape in self.bounds_shapes]
        offsets = [0] + list(it.accumulate(num_elems))[:-1]
        lbs = (
            self.in_lbs[..., offset : offset + size].reshape((-1, *shape))
            for offset, size, shape in strict_zip(
                offsets, num_elems, self.bounds_shapes
            )
        )
        ubs = (
            self.in_ubs[..., offset : offset + size].reshape((-1, *shape))
            for offset, size, shape in strict_zip(
                offsets, num_elems, self.bounds_shapes
            )
        )
        bound_args = (Box(lb, ub) for lb, ub in strict_zip(lbs, ubs))
        fixed_args = iter(self.fixed_args)
        args = tuple(
            next(bound_args) if has_bounds else next(fixed_args)
            for has_bounds in self.has_bounds
        )
        return args

    @classmethod
    def root_branch(
        cls,
        jaxpr: jax.extend.core.ClosedJaxpr,
        *args: Bounds | Real[Array, "..."],
        target: BAB_TARGET,
    ) -> "InputSplit":
        """Creates the branch data of the root branch."""
        has_bounds = tuple(is_bounds(arg) for arg in args)
        in_bounds = tuple(arg for arg in args if is_bounds(arg))
        fixed_args = tuple(arg for arg in args if not is_bounds(arg))

        in_lbs, in_ubs = strict_zip(*in_bounds)
        in_shapes = tuple(lb.shape for lb in in_lbs)

        in_lbs = jnp.concatenate([lbs.reshape(1, -1) for lbs in in_lbs], axis=-1)
        in_ubs = jnp.concatenate([ubs.reshape(1, -1) for ubs in in_ubs], axis=-1)
        return cls(in_lbs, in_ubs, fixed_args, in_shapes, has_bounds)

    def refine(
        self, idx: Int[Array, "b"], split_point: float | Real[Array, "b"] = 0.5
    ) -> "InputSplit":
        """Refines this input splitting into two new splits.

        Splits the input bounds along ``axis`` at ``split_point``.
        The new branches either have

            new_in_lbs[idx] = in_lbs[idx]
            new_in_ubs[idx] = (in_ubs[idx] - in_lbs[idx]) * split_point

        or
            new_in_lbs[idx] = (in_ubs[idx] - in_lbs[idx]) * split_point
            new_in_ubs[idx] = in_ubs[idx]

        All other input bounds remain unchanged.

        Args:
            idx: The axis to split for each batch element.
                Each element of ``idx`` needs to be a valid index of
                ``self.in_lbs`` and ``self.in_ubs``.
            split_point: Where to split the input bounds.
                Needs to have the same shape as ``idx``.
                Needs to be greater than ``0.0`` and less than ``1.0``.
                The default value of ``0.5`` splits at the center.
        Returns:
            The refined input splits.
        """
        # intentional use of np instead of jnp here to exclude assert when jitting
        assert np.all((0.0 < split_point) & (split_point < 1.0))
        assert self.in_lbs.ndim == 2
        assert self.in_ubs.ndim == 2

        idx = idx[:, None]
        lb_at = jnp.take_along_axis(self.in_lbs, idx, axis=1)
        ub_at = jnp.take_along_axis(self.in_ubs, idx, axis=1)
        split = (ub_at + lb_at) * split_point

        # left is [l, s] and right is [s, u] branch when splitting at s
        left_ubs = jnp.put_along_axis(self.in_ubs, idx, split, axis=1, inplace=False)
        right_lbs = jnp.put_along_axis(self.in_lbs, idx, split, axis=1, inplace=False)

        new_lbs = jnp.concatenate([self.in_lbs, right_lbs], axis=0)
        new_ubs = jnp.concatenate([left_ubs, self.in_ubs], axis=0)
        return InputSplit(
            new_lbs, new_ubs, self.fixed_args, self.bounds_shapes, self.has_bounds
        )


# ==============================================================================
# Compute Bounds
# ==============================================================================


class ComputeBoundsForInputSplitting(MakeForBaB[ComputeBoundsBaB[InputSplit]]):
    def __init__(
        self,
        compute_bounds: Callable[
            [jax.extend.core.ClosedJaxpr],
            Callable[[Bounds | Real[Array, "..."], ...], Bounds],
        ],
    ):
        self.make_compute_bounds = compute_bounds

    def __call__(
        self, jaxpr: jax.extend.core.ClosedJaxpr, *args, **__
    ) -> ComputeBoundsBaB[InputSplit]:
        # since jaxpr does not have a axis for branches, we need to add one
        # using vmap
        branch_axes = tuple(0 if is_bounds(arg) else None for arg in args)
        compute_bounds = jax.vmap(self.make_compute_bounds(jaxpr), in_axes=branch_axes)

        @functools.wraps(compute_bounds)
        def bounds_for_input_splitting(branches: InputSplit) -> Bounds:
            args = branches.as_input()
            return compute_bounds(*args)

        return bounds_for_input_splitting


crown_input_splitting = ComputeBoundsForInputSplitting(crown)
crown_ibp_input_splitting = ComputeBoundsForInputSplitting(crown_ibp)
ibp_input_splitting = ComputeBoundsForInputSplitting(ibp_jaxpr)


# ==============================================================================
# Splitting
# ==============================================================================


def split_longest_edge(*_, **__) -> SplitBranches[InputSplit]:
    """Splits branches along the longest edge.

    The longest edge is the input dimension with the largest diameter
    in the branch input space.

    For example, if the input space is three-dimensional and the input bounds
    are ``lb=[0, -1, 1], ub=[2, 2, 2]``, dimension one is selected, as it has
    the largest diameter of ``3``, while dimensions zero has diameter ``2``
    and dimension two has diameter ``1``.

    This is a factory function that creates a splitting procedure.
    """

    def split(branches: InputSplit) -> InputSplit:
        split_idx = longest_edge(branches.in_lbs, branches.in_ubs)
        return branches.refine(split_idx)

    return split


def longest_edge(
    in_lbs: Real[Array, "b d"], in_ubs: Real[Array, "b d"]
) -> Int[Array, "b"]:
    """Computes the longest edges of a batch.

    See `split_longest_edge` for more details.
    """
    edge_len = in_ubs - in_lbs
    return jnp.argmax(edge_len, axis=1)


# ==============================================================================
# Interface
# ==============================================================================


@fun_to_jaxpr_for_bab(bab_name="Input Splitting Branch and Bound")
def input_splitting_bab[BS: BranchSelection, AR: ArrayRef](
    jaxpr: jax.extend.core.ClosedJaxpr,
    select: MakeForBaB[SelectBranches[AR, BS]] = select_worst,
    split: MakeForBaB[SplitBranches["InputSplit"]] = split_longest_edge,
    compute_bounds: MakeForBaB[ComputeBoundsBaB["InputSplit"]] = ibp_input_splitting,
    target: typing.Literal["minimum", "maximum", "image"] | BAB_TARGET = "min",
    batch_size: int = 128,
    jit: bool = True,
    make_branch_store: Callable[
        ["InputSplit"], BranchStore["InputSplit", BS, AR]
    ] = SimpleBranchStore,
) -> BaBCallable:
    return bab(
        jaxpr,
        make_root_branch=InputSplit.root_branch,
        select=select,
        split=split,
        compute_bounds=compute_bounds,
        target=target,
        batch_size=batch_size,
        jit=jit,
        make_branch_store=make_branch_store,
    )
