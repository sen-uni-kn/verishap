# Copyright 2025 David Boetius
from dataclasses import dataclass
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np
from formalax import Box, crown_ibp, ibp
from jaxtyping import Array, Int, Real

from .branch_store import BranchStore
from .utils import argmax_k


@jax.tree_util.register_dataclass
@dataclass(eq=False, frozen=True)
class BranchData:
    # contains boolean masks but stored as float
    # for computing bounds
    coalitions: Box[Real[Array, " b *shape"]]
    contrib_lb: Real[Array, " b"]
    contrib_ub: Real[Array, " b"]
    depth: Int[Array, " b"]  # count from zero
    # sum of coalition weights in branch
    total_coalition_weight: Real[Array, " b"]
    # bounds on the Shapley value if this branch were the only branch
    shapley_lb: Real[Array, " b"]
    shapley_ub: Real[Array, " b"]


def contribution(
    value_fn: Callable[
        [Real[Array, " *shape"], Real[Array, " b *shape"]], Real[Array, " b"]
    ],
    x: Real[Array, " *shape"],
    feature: tuple[int, ...],
):
    """Computes the contribution of `feature` to coalitions.

    If `v` is the value function, the contribution is `v(S ∪ {i}) - v(S)`.

    Args:
        value_fn: The value function used to evaluate each coalition of features.
        x: The input feature values.
        feature: Index of the feature for which to compute the contribution.

    Returns:
        A function that computes the contribution of `feature` to coalitions.
    """
    include_feature = jnp.zeros_like(x)  # use float dtype for computing smears
    include_feature = include_feature.at[feature].set(1.0)

    def contrib(coalition: Real[Array, " b *shape"]):
        """Computes (v(S + {i}) - v(S))."""
        # these are floats, so we can't use bitwise_or
        one = jnp.ones_like(coalition)
        plus_feature = (coalition + include_feature).clip(max=one)

        with_feature = value_fn(x, plus_feature)
        without_feature = value_fn(x, coalition)
        return with_feature - without_feature

    return contrib


def shapley_bab(
    value_fn: Callable[
        [Real[Array, " *shape"], Real[Array, " b *shape"]], Real[Array, " b"]
    ],
    x: Real[Array, " *shape"],
    feature: tuple[int, ...],
    compute_bounds=crown_ibp,
    fast_compute_bounds=ibp,
    split_strategy: Literal["longest-edge", "strong-branching"] = "longest-edge",
    batch_size: int = 1024,
    jit: bool = True,
):
    """Compute and refine bounds on Shapley values.
    This function performs branch and bound on coalitions of input features.

        Representation of coalitions: This function represents coalitions as
            a boolean mask of the input features.
            Sets of coalitions are represented as bounds on the boolean mask.
            While this can not represent all sets of coalitions, it is a
            succinct representation that is easy to split and combine.

        Args:
            value_fn: The value function used to evaluate each coalition of features.
                The first argument of ``value_fn`` is are the input feature values.
                The second is a boolean mask, determining which input features
                are in the coalition.
                The output of ``value_fn`` is the value of the coalition.
            x: The input feature values.
            feature: Index of the feature for which to compute the Shapley value.
            split_strategy: The strategy to use for splitting branches.
            batch_size: The batch size to use for the branch and bound.
            jit: Whether to just-in-time compile the branch evaluation.

        Yields:
            Bound on the Shapley value of the feature.
    """
    # --------------------------------------------------------------------------
    # Abbreviations:
    # - lb: lower bound
    # - ub: upper bound
    # - contrib: contribution
    # - coali: coalition
    # - coaliw: coalition weight
    # - val: value
    # - diff: difference
    # --------------------------------------------------------------------------
    data_axes = tuple(range(1, x.ndim + 1))

    contrib = contribution(value_fn, x, feature)
    bound_contrib = compute_bounds(contrib)

    def select(branches: BranchData):
        score = branches.shapley_ub - branches.shapley_lb

        select_size = min(batch_size, len(score))
        return argmax_k(score, select_size)

    def split(branches: BranchData):
        """Split branches by including/excluding one feature."""
        coali_lb, coali_ub = branches.coalitions
        num_branches, *in_shape = coali_lb.shape

        # grad_lbs, grad_ubs = compute_smears(branches.coalitions)
        # smears = jnp.maximum(jnp.abs(grad_lbs), jnp.abs(grad_ubs))
        # smears = jnp.reshape(smears, (smears.shape[0], -1))
        # # already fixed features have smears of 0
        # split_axis = jnp.argmax(smears, axis=1)

        coali_lb_ = jnp.reshape(coali_lb, (num_branches, -1))
        coali_ub_ = jnp.reshape(coali_ub, (num_branches, -1))

        match split_strategy:
            case "longest-edge":
                edge_len = jnp.abs((coali_ub_ - coali_lb_) * x)
                split_axis = jnp.argmax(edge_len, axis=-1)
            case "strong-branching":
                pass
            case _:
                raise ValueError(f"Invalid split strategy: {split_strategy}")

        left_ub = coali_ub_.at[np.arange(num_branches), split_axis].set(0.0)
        right_lb = coali_lb_.at[np.arange(num_branches), split_axis].set(1.0)
        left_ub = jnp.reshape(left_ub, coali_lb.shape)
        right_lb = jnp.reshape(right_lb, coali_lb.shape)

        old_depth = branches.depth
        old_num_included = coali_lb_.sum(axis=-1)
        old_total_coaliw = branches.total_coalition_weight
        # left branch is exclude branch
        left_total_coaliw = (
            (old_depth + 1 - old_num_included) / (old_depth + 2) * old_total_coaliw
        )
        right_total_coaliw = (old_num_included + 1) / (old_depth + 2) * old_total_coaliw

        new_coali_lb = jnp.concat([coali_lb, right_lb])
        new_coali_ub = jnp.concat([left_ub, coali_ub])
        new_total_coaliw = jnp.concat([left_total_coaliw, right_total_coaliw])

        return Box(new_coali_lb, new_coali_ub), new_total_coaliw

    def compute_bounds(
        coalitions: Box[Real[Array, " b *shape"]],
        total_coaliw: Real[Array, " b"],
    ):
        """Compute value and branch-local Shapley value bounds."""
        contrib_lbs, contrib_ubs = bound_contrib(coalitions).concrete

        # total coaliw is the sum of all coalitions weights in the branch
        shapley_lbs = total_coaliw * contrib_lbs
        shapley_ubs = total_coaliw * contrib_ubs
        return contrib_lbs, contrib_ubs, shapley_lbs, shapley_ubs

    def bab_step(batch: BranchData, num_branches: int):
        new_coalitions, new_total_coaliw = split(batch)
        new_depths = batch.depth + 1
        new_depths = jnp.concat([new_depths, new_depths], axis=0)
        new_contrib_lbs, new_contrib_ubs, new_shapley_lbs, new_shapley_ubs = compute_bounds(
            new_coalitions, new_total_coaliw
        )
        return BranchData(
            new_coalitions,
            new_contrib_lbs,
            new_contrib_ubs,
            new_depths,
            new_total_coaliw,
            new_shapley_lbs,
            new_shapley_ubs,
        )

    if jit:
        bab_step_jit = jax.jit(bab_step, static_argnums=(1,))

        def bab_step(batch: BranchData, num_branches: int):
            padding = batch_size - num_branches

            def pad(x):
                pad_val = jnp.empty((padding, *x.shape[1:]), dtype=x.dtype)
                return jnp.concat([x, pad_val], axis=0)

            def unpad(x_padded):
                # the values that we unpad have twice the batch size
                # since they contain first the left branch and
                # then the right branches
                left = x_padded[:num_branches]
                right = x_padded[batch_size : batch_size + num_branches]
                return jnp.concat([left, right], axis=0)

            padded = jax.tree.map(pad, batch)
            out_padded = bab_step_jit(padded, num_branches=batch_size)
            out = jax.tree.map(unpad, out_padded)
            return out

    # Root branch
    without_feature = jnp.ones_like(x).at[feature].set(0.0)
    all_coalitions = Box(
        jnp.zeros((1,) + x.shape, dtype=x.dtype),
        jnp.expand_dims(without_feature, axis=0),
    )
    zero_depth = jnp.zeros((1,), dtype=int)
    total_coaliw = jnp.ones((1,), dtype=x.dtype)
    contrib_lb, contrib_ub, shapley_lb, shapley_ub = compute_bounds(
        all_coalitions, total_coaliw
    )
    root_data = BranchData(
        all_coalitions, contrib_lb, contrib_ub, zero_depth, total_coaliw, shapley_lb, shapley_ub
    )

    branches: BranchStore = BranchStore(root_data)
    shapley_lb, shapley_ub = root_data.shapley_lb, root_data.shapley_ub
    while True:
        # Prune completely split branches
        coali_lb, coali_ub = branches.data.coalitions
        single_coalition = (coali_lb == coali_ub).all(axis=data_axes)
        # Also prune branches with tight value bounds
        tight_bounds = jnp.isclose(branches.data.contrib_ub, branches.data.contrib_lb)
        # shapley bounds from the pruned branches remain part of the global bounds
        branches.pop(single_coalition | tight_bounds)

        yield Box(shapley_lb.squeeze(), shapley_ub.squeeze())
        if len(branches) == 0 or jnp.isclose(shapley_lb, shapley_ub):
            return None

        selected = select(branches.data)
        batch = branches.pop(selected)
        # Subtract here, since we will refine the bounds for batch
        shapley_lb = shapley_lb - batch.shapley_lb.sum()
        shapley_ub = shapley_ub - batch.shapley_ub.sum()

        new_branches = bab_step(batch, selected.sum())
        branches.add(new_branches)
        shapley_lb = shapley_lb + new_branches.shapley_lb.sum()
        shapley_ub = shapley_ub + new_branches.shapley_ub.sum()
