# Copyright 2025 David Boetius
import math
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
from formalax import Box, ibp
from formalax.verify.bab.branch_store import (
    BranchStore,
    MaskBranchSelection,
    SimpleBranchStore,
)
from jaxtyping import Array, Int, Real
from scipy.special import comb


@jax.tree_util.register_dataclass
@dataclass(eq=False, frozen=True)
class BranchData:
    # contains boolean masks but stored as float
    # for computing gradients
    coalitions: Box[Real[Array, " b *shape"]]
    val_lb: Real[Array, " b"]
    val_ub: Real[Array, " b"]
    depth: Int[Array, " b"]  # count from zero
    # bounds on the Shapley value if each branch
    # were the only branch
    shapley_lb: Real[Array, " b"]
    shapley_ub: Real[Array, " b"]


def bound_coalition_weights(
    num_features: int,
) -> Callable[[Box[Real[Array, " b *shape"]]], Box[Real[Array, " b"]]]:
    """Compute bounds on the weight of a coalition.

    Compute bounds on
    c(S) = |S|! * (|N| - |S| - 1)! / |N|!
         = 1/|N| * 1/C(|N| - 1, |S|),
    where C(n, k) is the binomial coefficient.
    Since the binomial coefficient is concave,
    c(S) is convex.
    """

    # FIXME: exact=True is slow. Use?
    combs = [comb(num_features, i, exact=True) for i in range(num_features)]
    # do this in Python because the integers can be too large for int32
    coalition_weights = [1 / (num_features * comb) for comb in combs]
    coalition_weights = jnp.array(coalition_weights)
    mid_size = num_features // 2
    min_coalition_weight = coalition_weights[mid_size]

    def bound_coalition_weight(
        coalitions: Box[Real[Array, " b *shape"]],
    ) -> Box[Real[Array, " b"]]:
        # The coalitions set is represented as bounds on the boolean mask.
        # The smallest coalition in the set excludes all features that have
        # a lower bound on 0 in coalitions.
        # Therefore, summing out the lower bound of the mask gives
        # the size of the smallest coalition.
        # Similarly for the largest coalition.
        data_axes = tuple(range(1, coalitions.lower_bound.ndim))
        size_lb = coalitions.lower_bound.sum(axis=data_axes).astype(jnp.int32)
        size_ub = coalitions.upper_bound.sum(axis=data_axes).astype(jnp.int32)

        lb = jnp.where(
            (size_ub < mid_size),
            coalition_weights[size_ub],  # in falling part of weights
            jnp.where(
                (size_lb > mid_size),
                coalition_weights[size_lb],  # in rising part of weights
                min_coalition_weight,  # minimum included in [size_lb, size_ub]
            ),
        )
        ub = jnp.where(
            size_lb < num_features - size_ub,
            coalition_weights[size_lb],
            coalition_weights[size_ub],
        )

        return Box(lb, ub)

    return bound_coalition_weight


def value_difference(
    value_fn: Callable[
        [Real[Array, " *shape"], Real[Array, " b *shape"]], Real[Array, " b"]
    ],
    x: Real[Array, " *shape"],
    feature: tuple[int, ...],
):
    include_feature = jnp.zeros_like(x)  # use float dtype for computing smears
    include_feature = include_feature.at[feature].set(1.0)

    def val_difference(coalition: Real[Array, " b *shape"]):
        """Computes (v(S + {i}) - v(S))."""
        # these are floats, so we can't use bitwise_or
        one = jnp.ones_like(coalition)
        plus_feature = (coalition + include_feature).clip(max=one)

        with_feature = value_fn(x, plus_feature)
        without_feature = value_fn(x, coalition)
        return (with_feature - without_feature).squeeze()

    return val_difference


def shapley_bab(
    value_fn: Callable[
        [Real[Array, " *shape"], Real[Array, " b *shape"]], Real[Array, " b"]
    ],
    x: Real[Array, " *shape"],
    feature: tuple[int, ...],
    compute_bounds=ibp,
    fast_compute_bounds=ibp,
    batch_size: int = 1024,
    jit: bool = True,
    make_branch_store: Callable[[Any], BranchStore] = SimpleBranchStore,
):
    """Compute and refine bounds on Shapley values.
    This function performs branch and bound on coalitions of input features.

        Representation of coalitions: This function represents coalitions as
            a boolean mask of the input features.
            Sets of coalitions are represented as bounds on the boolean mask.
            While this can not represent all sets of coalitions, it is a succint
            representation that is easy to split and combine.

        Args:
            value_fn: The value function used to evaluate each coalition of features.
                The first argument of ``value_fn`` is are the input feature values.
                The second is a boolean mask, determining which input features
                are in the coalition.
                The output of ``value_fn`` is the value of the coalition.
            x: The input feature values.
            feature: Index of the feature for which to compute the Shapley value.
            batch_size: The batch size to use for the branch and bound.
            jit: Whether to just-in-time compile the branch evaluation.
            make_branch_store: A function that creates a branch store.

        Yields:
            Bound on the Shapley value of the feature.
    """
    # --------------------------------------------------------------------------
    # Abbreviations:
    # - lb: lower bound
    # - ub: upper bound
    # - coali: coalition
    # - coaliw: coalition weight
    # - val: value
    # - diff: difference
    # --------------------------------------------------------------------------

    num_features = math.prod(x.shape)
    data_axes = tuple(range(1, x.ndim + 1))
    # Computes all binomial coefficients up to num_features.
    # This can take a few seconds.
    bound_coali_weights = bound_coalition_weights(num_features)

    val_diff = value_difference(value_fn, x, feature)
    bound_val_diff = compute_bounds(jax.vmap(val_diff))

    def select(branches: BranchData):
        score = branches.shapley_ub - branches.shapley_lb

        select_size = min(batch_size, len(score))
        return score.argmax_k(select_size)

    # Compute smears for splitting.
    # Smears are bounds on the gradient of a function.
    # val_diff_grads = jax.vmap(jax.grad(val_diff))
    # compute_smears = fast_compute_bounds(val_diff_grads)

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
        # FIXME: this just splits features in order
        split_axis = jnp.argmax(coali_ub_ - coali_lb_, axis=1)

        left_ub = coali_ub_.at[:, split_axis].set(0.0)
        right_lb = coali_lb_.at[:, split_axis].set(1.0)
        left_ub = jnp.reshape(left_ub, coali_lb.shape)
        right_lb = jnp.reshape(right_lb, coali_lb.shape)

        new_coali_lb = jnp.concat([coali_lb, right_lb])
        new_coali_ub = jnp.concat([left_ub, coali_ub])
        return Box(new_coali_lb, new_coali_ub)

    def compute_bounds(
        coalitions: Box[Real[Array, " b *shape"]], depths: Int[Array, " b"]
    ):
        """Compute value and branch-local Shapley value bounds."""
        val_lbs, val_ubs = bound_val_diff(coalitions).concrete
        coaliw_lb, coaliw_ub = bound_coali_weights(coalitions)

        summand_lbs = jnp.where(val_lbs > 0, coaliw_lb, coaliw_ub) * val_lbs
        summand_ubs = jnp.where(val_ubs < 0, coaliw_lb, coaliw_ub) * val_ubs

        # -1 because the feature we compute the Shapley value for
        # is also excluded
        num_coalitions = 2 ** (num_features - depths - 1)
        shapley_lbs = num_coalitions * summand_lbs
        shapley_ubs = num_coalitions * summand_ubs
        return val_lbs, val_ubs, shapley_lbs, shapley_ubs

    def bab_step(batch, num_branches: int):
        new_coalitions = split(batch)
        new_depths = batch.depth + 1
        new_depths = jnp.concat([new_depths, new_depths], axis=0)
        new_val_lbs, new_val_ubs, new_shapley_lbs, new_shapley_ubs = compute_bounds(
            new_coalitions, new_depths
        )
        return BranchData(
            new_coalitions,
            new_val_lbs,
            new_val_ubs,
            new_depths,
            new_shapley_lbs,
            new_shapley_ubs,
        )

    if jit:
        bab_step_jit = jax.jit(bab_step)
        # FIXME: pad inputs to branch_size
        bab_step_no_jit = bab_step

        def bab_step(batch, num_branches: int):
            if num_branches < batch_size:
                # avoid recompilation when there are few branches
                return bab_step_no_jit(batch, num_branches=num_branches)
            else:
                return bab_step_jit(batch, num_branches=num_branches)

    def shapley_bounds(
        branches: BranchData,
    ):
        """Computes bounds on the overall Shapley value."""
        # FIXME: should not access data (specific for SimpleBranchStore)
        shapley_lb = branches.shapley_lb.data.sum()
        shapley_ub = branches.shapley_ub.data.sum()
        return shapley_lb, shapley_ub

    # Root branch
    all_coalitions = Box(
        jnp.zeros((1,) + x.shape, dtype=x.dtype),
        jnp.ones((1,) + x.shape, dtype=x.dtype),
    )
    zero_depth = jnp.zeros((1,), dtype=int)
    val_lb, val_ub, shapley_lb, shapley_ub = compute_bounds(all_coalitions, zero_depth)
    root_data = BranchData(
        all_coalitions, val_lb, val_ub, zero_depth, shapley_lb, shapley_ub
    )

    branches: BranchStore = make_branch_store(root_data)
    pruned_shapley_lb = pruned_shapley_ub = jnp.zeros((), dtype=val_lb.dtype)
    while True:
        shapley_lb, shapley_ub = shapley_bounds(branches.data)
        shapley_lb = shapley_lb + pruned_shapley_lb
        shapley_ub = shapley_ub + pruned_shapley_ub
        yield Box(shapley_lb, shapley_ub)

        if len(branches) == 0:
            # All branches are completely split
            return None

        # Prune completely split branches
        coali_lb, coali_ub = branches.data.coalitions.concrete
        # FIXME: should not access data (specific for SimpleBranchStore)
        single_coalition = (coali_lb.data == coali_ub.data).all(axis=data_axes)
        single_coalition = MaskBranchSelection(branches, single_coalition)
        pruned = branches.pop(single_coalition)
        pruned_shapley_lb = pruned_shapley_lb + pruned.shapley_lb.sum()
        pruned_shapley_ub = pruned_shapley_ub + pruned.shapley_ub.sum()

        selected = select(branches.data)
        batch = branches.pop(selected)

        new_branches = bab_step(batch, len(selected))
        branches.add(new_branches)
