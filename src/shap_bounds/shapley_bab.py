# Copyright 2025 David Boetius
import itertools as it
from dataclasses import dataclass
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np
from formalax import Box, crown_ibp, ibp
from jaxtyping import Array, Int, Real

from .branch_store import BranchStore
from .utils import argmax_k, argmin_k


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
    base_mask: Real[Array, " *shape"],
    feature: tuple[int, ...],
):
    """Computes the contribution of `feature` to coalitions.

    If `v` is the value function, the contribution is `v(S ∪ {i}) - v(S)`.

    Args:
        value_fn: The value function used to evaluate each coalition of features.
        base_mask: A base mask as an input to the value function.
            Can have arbitrary values but needs to have the correct shape.
        feature: Index of the feature for which to compute the contribution.

    Returns:
        A function that computes the contribution of `feature` to coalitions.
    """
    include_feature = jnp.zeros_like(base_mask)  # use float dtypes for computing bounds
    include_feature = include_feature.at[feature].set(1.0)

    def contrib(coalition: Real[Array, " b *shape"]):
        """Computes (v(S + {i}) - v(S))."""
        # these are floats, so we can't use bitwise_or
        one = jnp.ones_like(coalition)
        plus_feature = (coalition + include_feature).clip(max=one)

        with_feature = value_fn(plus_feature)
        without_feature = value_fn(coalition)
        return with_feature - without_feature

    return contrib


def shapley_bab(
    value_fn: Callable[
        [Real[Array, " *shape"], Real[Array, " b *shape"]], Real[Array, " b"]
    ],
    base_mask: Real[Array, " *shape"],
    feature: tuple[int, ...],
    compute_bounds=crown_ibp,
    fast_compute_bounds=ibp,
    select_strategy: Literal["max-diam", "min-diam", "first"] = "max-diam",
    split_strategy: Literal[
        "longest-edge",
        "strong-branching-better",
        "strong-branching-worse",
        "smart-branching-ibp-better",
        "smart-branching-ibp-worse",
    ] = "smart-branching-ibp-worse",
    batch_size: int = 1024,
    jit: bool = True,
    log: bool = True,
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
            base_mask: A base mask as an input to the value function.
                Can have arbitrary values but needs to have the correct shape.
            feature: Index of the feature for which to compute the Shapley value.
            select_strategy: The strategy to use for selecting branches.
                - "max-diam": Select the branch with the largest difference between upper and lower bound.
                - "min-diam": Select the branch with the smallest difference between upper and lower bound.
            split_strategy: The strategy to use for splitting branches.
            batch_size: The batch size to use for the branch and bound.
            jit: Whether to just-in-time compile the branch evaluation.
            log: Whether to print status messages.

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
    data_axes = tuple(range(1, base_mask.ndim + 1))

    contrib = contribution(value_fn, base_mask, feature)
    bound_contrib = compute_bounds(contrib)
    fast_bound_contrib = fast_compute_bounds(contrib)

    def select(branches: BranchData):
        num_branches = len(branches.shapley_ub)
        select_size = min(batch_size, num_branches)

        if select_strategy == "first":
            indices = jnp.arange(select_size)
            return jnp.zeros((num_branches,), dtype=bool).at[indices].set(True)

        score = branches.shapley_ub - branches.shapley_lb
        match select_strategy:
            case "max-diam":
                return argmax_k(score, select_size)
            case "min-diam":
                return argmin_k(score, select_size)
            case _:
                raise ValueError(f"Invalid select strategy: {select_strategy}")

    def compute_bounds(
        coalitions: Box[Real[Array, " b *shape"]],
        total_coaliw: Real[Array, " b"],
        bound_contrib=bound_contrib,
    ):
        """Compute value and branch-local Shapley value bounds."""
        contrib_lbs, contrib_ubs = bound_contrib(coalitions).concrete

        # total coaliw is the sum of all coalitions weights in the branch
        shapley_lbs = total_coaliw * contrib_lbs
        shapley_ubs = total_coaliw * contrib_ubs
        return contrib_lbs, contrib_ubs, shapley_lbs, shapley_ubs

    def split(
        branches: BranchData, split_axes: Int[Array, " b"]
    ) -> tuple[Box[Real[Array, " b *shape"]], Real[Array, " b"]]:
        """Split branches by including/excluding one feature.

        Args:
            branches: The branches to split.
            split_axes: The axes to split on in the flattened coalition mask.

        Returns:
            The coalition bounds and the total coalition weights after splitting.
        """
        coali_lb, coali_ub = branches.coalitions
        num_branches = coali_lb.shape[0]
        coali_lb_ = jnp.reshape(coali_lb, (num_branches, -1))
        coali_ub_ = jnp.reshape(coali_ub, (num_branches, -1))

        left_ub = coali_ub_.at[np.arange(num_branches), split_axes].set(0.0)
        right_lb = coali_lb_.at[np.arange(num_branches), split_axes].set(1.0)
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

    def select_split(branches: BranchData):
        """Selects one feature to split per branch."""
        coali_lb, coali_ub = branches.coalitions

        num_branches = coali_lb.shape[0]
        coali_lb_ = jnp.reshape(coali_lb, (num_branches, -1))
        coali_ub_ = jnp.reshape(coali_ub, (num_branches, -1))

        if split_strategy == "longest-edge":
            edge_len = jnp.abs((coali_ub_ - coali_lb_) * x)
            split_axes = jnp.argmax(edge_len, axis=-1)
        elif split_strategy.startswith("strong-branching") or split_strategy.startswith(
            "smart-branching-ibp"
        ):
            bound_method = (
                bound_contrib
                if split_strategy.startswith("strong-branching")
                else fast_bound_contrib
            )
            num_features = coali_lb_.shape[-1]

            def eval_split(carry, i: Int[Array, ""]):
                i_array = jnp.full((num_branches,), i, dtype=int)
                split_coalis, split_total_coaliw = split(branches, i_array)
                _, _, split_lbs, split_ubs = compute_bounds(
                    split_coalis,
                    split_total_coaliw,
                    bound_contrib=bound_method,
                )
                diameter = split_ubs - split_lbs
                if split_strategy.endswith("-better"):
                    # here we look at the *smaller* diameter hoping for fast pruning
                    score = diameter.reshape(2, num_branches).min(axis=0)
                elif split_strategy.endswith("-worse"):
                    # here we look at the *larger* diameter hoping for tighter bounds
                    score = diameter.reshape(2, num_branches).max(axis=0)
                else:
                    raise ValueError(f"Invalid split strategy: {split_strategy}")
                is_valid = coali_lb_[:, i] != coali_ub_[:, i]
                return carry, jnp.where(is_valid, score, jnp.inf)

            _, split_scores = jax.lax.scan(eval_split, None, jnp.arange(num_features))
            split_axes = jnp.argmin(split_scores, axis=0)
        else:
            raise ValueError(f"Invalid split strategy: {split_strategy}")

        # grad_lbs, grad_ubs = compute_smears(branches.coalitions)
        # smears = jnp.maximum(jnp.abs(grad_lbs), jnp.abs(grad_ubs))
        # smears = jnp.reshape(smears, (smears.shape[0], -1))
        # # already fixed features have smears of 0
        # split_axes = jnp.argmax(smears, axis=1)
        return split_axes

    def bab_step(batch: BranchData, num_branches: int):
        split_axes = select_split(batch)
        new_coalitions, new_total_coaliw = split(batch, split_axes)
        new_contrib_lbs, new_contrib_ubs, new_shapley_lbs, new_shapley_ubs = (
            compute_bounds(new_coalitions, new_total_coaliw)
        )
        new_depths = jnp.concat([batch.depth + 1, batch.depth + 1], axis=0)
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
    without_feature = jnp.ones_like(base_mask).at[feature].set(0.0)
    all_coalitions = Box(
        jnp.zeros((1,) + base_mask.shape, dtype=base_mask.dtype),
        jnp.expand_dims(without_feature, axis=0),
    )
    zero_depth = jnp.zeros((1,), dtype=int)
    total_coaliw = jnp.ones((1,), dtype=base_mask.dtype)
    contrib_lb, contrib_ub, shapley_lb, shapley_ub = compute_bounds(
        all_coalitions, total_coaliw
    )
    root_data = BranchData(
        all_coalitions,
        contrib_lb,
        contrib_ub,
        zero_depth,
        total_coaliw,
        shapley_lb,
        shapley_ub,
    )

    branches: BranchStore = BranchStore(root_data)
    shapley_lb, shapley_ub = root_data.shapley_lb, root_data.shapley_ub
    for i in it.count():
        # Prune completely split branches
        coali_lb, coali_ub = branches.data.coalitions
        single_coalition = (coali_lb == coali_ub).all(axis=data_axes)
        # Also prune branches with tight value bounds
        tight_bounds = jnp.isclose(branches.data.contrib_ub, branches.data.contrib_lb)
        # shapley bounds from the pruned branches remain part of the global bounds
        branches.pop(single_coalition | tight_bounds)

        yield Box(shapley_lb.squeeze(), shapley_ub.squeeze())

        if log:
            num_branches = len(branches)
            num_fully_split = single_coalition.sum()
            num_tight = (tight_bounds & ~single_coalition).sum()
            lb, ub = shapley_lb.item(), shapley_ub.item()
            mid, ran = (lb + ub) / 2, (ub - lb) / 2
            print(
                f"[i: {i:3d}] S ∈ [{mid:8.4f} ± {ran:8.4f}]\t|"
                f" {num_branches} branches, pruned: {num_tight} tight, {num_fully_split} fully split"
            )

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
