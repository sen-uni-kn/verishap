# Copyright 2025 David Boetius
import itertools as it
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np
from formalax import Box, alpha_crown, crown, crown_ibp, ibp
from jaxtyping import Array, Bool, Int, Real

from .logger import ConsoleLogger, Logger
from .priority_branch_store import PriorityBranchStore
from .timer import Timer


@jax.tree_util.register_dataclass
@dataclass(eq=False, frozen=True)
class BranchData:
    # contains boolean masks but stored as float
    # for computing bounds
    coalitions: Box[Real[Array, " b *shape"]]
    value_lb: Real[Array, " b"]
    value_ub: Real[Array, " b"]
    num_splits: Int[Array, " b"]  # count from zero
    # Sum of coalition weights in branch.
    # The sum of coalition weights is the same indepedently of
    # the feature we are computing the Shapley value for.
    total_coalition_weight: Real[Array, " b"]
    branch_index: Int[Array, " b"]  # count from zero


def smears(
    value_fn: Callable[[Real[Array, " b *shape"]], Real[Array, " b"]],
    compute_bounds=ibp,
):
    def single_value(coalition: Real[Array, " *shape"]) -> Real[Array, ""]:
        coalitions = jnp.expand_dims(coalition, axis=0)
        return value_fn(coalitions).squeeze()

    value_grads: Callable[[Real[Array, " b *shape"]], Real[Array, " b *shape"]] = (
        jax.vmap(jax.grad(single_value))
    )
    smears = compute_bounds(value_grads)
    return smears


def multi_shap_bab(
    value_fn: Callable[[Real[Array, " b *shape"]], Real[Array, " b"]],
    base_mask: Real[Array, " *shape"],
    features: Sequence[tuple[int, ...]] | None = None,
    compute_bounds: Callable
    | Literal["crown_ibp", "ibp", "crown", "alpha-crown"] = crown_ibp,
    fast_compute_bounds=ibp,
    select_strategy: Literal["max-diam", "min-diam", "fifo", "lifo"] = "max-diam",
    split_strategy: Literal[
        "longest-edge",
        "smears",
        "lirpa-weights",
        "strong-branching-better",
        "strong-branching-worse",
        "smart-branching-ibp-better",
        "smart-branching-ibp-worse",
    ] = "smart-branching-ibp-worse",
    batch_size: int = 1024,
    jit: bool = True,
    log: Logger | bool = True,
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
            features: Indices of the features for which to compute the Shapley values.
                If ``None``, computes the Shapley values of all features.
            select_strategy: The strategy to use for selecting branches.
                - "max-diam": Select the branch with the largest difference between
                              upper and lower bound.
                - "min-diam": Select the branch with the smallest difference between
                              upper and lower bound.
            split_strategy: The strategy to use for splitting branches.
            batch_size: The batch size to use for the branch and bound.
            jit: Whether to just-in-time compile the branch evaluation.
            log: Logger to log status messages.
                If ``True``, a ``ConsoleLogger`` is used.
                If ``False``, no logging is done.
                If a ``Logger`` is provided, it is used to log the status messages.

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
    if isinstance(compute_bounds, str):
        match compute_bounds:
            case "crown_ibp":
                compute_bounds = crown_ibp
            case "ibp":
                compute_bounds = ibp
            case "crown":
                compute_bounds = crown
            case "alpha-crown":
                compute_bounds = alpha_crown

    if log is True:
        log = ConsoleLogger()

    if features is None:
        features = jnp.indices(base_mask.shape).reshape(base_mask.ndim, -1).T
        features = [tuple(f) for f in features.tolist()]
    elif isinstance(features, int):
        features = [features]

    if log is not False:
        log.log_config(
            "multi_shap_bab",
            features=features,
            compute_bounds=compute_bounds.__name__,
            fast_compute_bounds=fast_compute_bounds.__name__,
            select_strategy=select_strategy,
            split_strategy=split_strategy,
            batch_size=batch_size,
            jit=jit,
        )
    timer = Timer()

    data_axes = tuple(range(1, base_mask.ndim + 1))
    bound_value = compute_bounds(value_fn)
    fast_bound_value = fast_compute_bounds(value_fn)
    value_smears = smears(value_fn)

    def selection_priority(branches: BranchData):
        """Computes the priority of branches that determine the order of selection."""
        match select_strategy:
            case "fifo" | "lifo":
                scores = branches.branch_index
                return scores if select_strategy == "lifo" else -scores
            case "max-diam" | "min-diam":
                total_coaliw = branches.total_coalition_weight
                diameters = (branches.value_ub - branches.value_lb) * total_coaliw
                return diameters if select_strategy == "max-diam" else -diameters
            case _:
                raise ValueError(f"Invalid select strategy: {select_strategy}")

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

        s, r = branches.num_splits, coali_lb_.sum(axis=-1)
        old_total_coaliw = branches.total_coalition_weight
        # left branch is exclude branch
        left_total_coaliw = (s + 1 - r) / (s + 2) * old_total_coaliw
        right_total_coaliw = (r + 1) / (s + 2) * old_total_coaliw

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
            edge_len = jnp.abs(coali_ub_ - coali_lb_)
            split_axes = jnp.argmax(edge_len, axis=-1)
        elif split_strategy == "smears":
            grad_lbs, grad_ubs = value_smears(branches.coalitions).concrete
            smears_ = jnp.maximum(jnp.abs(grad_lbs), jnp.abs(grad_ubs))
            smears_ = smears_ * jnp.abs(coali_ub_ - coali_lb_)
            split_axes = jnp.argmax(smears_, axis=-1)
        elif split_strategy == "lirpa-weights":
            lirpa_bounds = bound_value(branches.coalitions)
            lb_weights = lirpa_bounds.lb_weights[0]
            ub_weights = lirpa_bounds.ub_weights[0]
            influence = jnp.maximum(jnp.abs(lb_weights), jnp.abs(ub_weights))
            influence = influence * jnp.abs(coali_ub_ - coali_lb_)
            split_axes = jnp.argmax(influence, axis=-1)
        elif split_strategy.startswith("strong-branching") or split_strategy.startswith(
            "smart-branching-ibp"
        ):
            compute_bounds = (
                bound_value
                if split_strategy.startswith("strong-branching")
                else fast_bound_value
            )
            num_features = coali_lb_.shape[-1]

            def eval_split(carry, i: Int[Array, ""]):
                i_array = jnp.full((num_branches,), i, dtype=int)
                split_coalis, _ = split(branches, i_array)
                val_lbs, val_ubs = compute_bounds(split_coalis).concrete
                diameter = val_ubs - val_lbs
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

        return split_axes

    def to_prune(
        branches: BranchData,
    ) -> tuple[Bool[Array, " b"], Bool[Array, " b"], Bool[Array, " b"]]:
        """These steps can be awefully slow for some reason if not jitted."""
        # Prune completely split branches
        coali_lb, coali_ub = branches.coalitions
        single_coalition = (coali_lb == coali_ub).all(axis=data_axes)
        # Also prune branches with tight value bounds
        tight_bounds = jnp.isclose(branches.value_ub, branches.value_lb)
        prune = single_coalition | tight_bounds
        return prune, single_coalition, tight_bounds

    # Compute SHAP bounds from value bounds
    zeros = jnp.zeros_like(base_mask, dtype=float)
    ones = jnp.ones_like(base_mask, dtype=float)
    with_features: Real[Array, " f 1 *n"] = jnp.stack(
        [jnp.expand_dims(zeros.at[i].set(1.0), axis=0) for i in features]
    )
    without_features: Real[Array, " f 1 *n"] = jnp.stack(
        [jnp.expand_dims(ones.at[i].set(0.0), axis=0) for i in features]
    )
    data_axes2 = tuple(ax + 1 for ax in data_axes)

    def shap_bounds(branches: BranchData) -> Box:
        coali_lb, coali_ub = branches.coalitions
        val_lb, val_ub = branches.value_lb, branches.value_ub

        # branches where all sets contain the feature (i)
        with_i: Real[Array, " f b"]
        with_i = (coali_lb >= with_features).all(axis=data_axes2)
        # branches where all sets do not contain the feature
        without_i = (coali_ub <= without_features).all(axis=data_axes2)
        # branches where some sets contain the feature and some do not
        both = ~(with_i | without_i)

        # The coalition weights in branches are for all features, while
        # we need coalition weights for one feature (i) excluded.
        # The coalition weights for the with_i branches include one
        # feature too much, while the coalition weights for the
        # without_i branches exclude one feature too much.
        # The "both" branches already have the correct coalition weights.
        s, r = branches.num_splits, coali_lb.sum(axis=data_axes)
        sum_coaliw = branches.total_coalition_weight
        # Reduce number of included features by one
        sum_coaliw_with_i = (s + 1) / r * sum_coaliw
        # Reduce number of excluded features by one
        sum_coaliw_without_i = (s + 1) / (s - r) * sum_coaliw

        # Computing bounds on: sum lambda * v(S + {i}) - sum lambda * v(S)
        contrib_lb = (
            sum_coaliw_with_i * val_lb * with_i
            + sum_coaliw * val_lb * both
            - sum_coaliw * val_ub * both
            - sum_coaliw_without_i * val_ub * without_i
        )  # .sum(axis=-1)  # sum outside jitted code to handle padding correctly
        contrib_ub = (
            sum_coaliw_with_i * val_ub * with_i
            + sum_coaliw * val_ub * both
            - sum_coaliw * val_lb * both
            - sum_coaliw_without_i * val_lb * without_i
        )  # .sum(axis=-1)
        contrib_lb: Real[Array, " b f"] = jnp.moveaxis(contrib_lb, -1, 0)
        contrib_ub = jnp.moveaxis(contrib_ub, -1, 0)
        return Box(contrib_lb, contrib_ub)

    def bab_step(
        batch: BranchData, num_branches: int
    ) -> tuple[
        BranchData,
        Real[Array, ""],
        Real[Array, ""],
        tuple[Bool[Array, " b"], Bool[Array, " b"], Bool[Array, " b"]],
    ]:
        old_shap_bounds = shap_bounds(batch)

        split_axes = select_split(batch)
        new_coalitions, new_total_coaliw = split(batch, split_axes)
        new_value_lbs, new_value_ubs = bound_value(new_coalitions).concrete
        new_num_splits = jnp.concat(
            [batch.num_splits + 1, batch.num_splits + 1], axis=0
        )
        # Count branches as in a binary heap
        new_branch_index = jnp.concat(
            [2 * batch.branch_index + 1, 2 * batch.branch_index + 2], axis=0
        )
        new_branches = BranchData(
            new_coalitions,
            new_value_lbs,
            new_value_ubs,
            new_num_splits,
            new_total_coaliw,
            new_branch_index,
        )

        new_shap_bounds = shap_bounds(new_branches)
        return new_branches, (old_shap_bounds, new_shap_bounds), to_prune(new_branches)

    def drop_pruned(
        prune: Bool[Array, " b"], array: Real[Array, " b *shape"]
    ) -> Real[Array, " c *shape"]:
        return array[~prune]

    if jit:
        bab_step_jit = jax.jit(bab_step, static_argnums=(1,))

        def pad(x, padding: int):
            pad_val = jnp.empty((padding, *x.shape[1:]), dtype=x.dtype)
            return jnp.concat([x, pad_val], axis=0)

        def unpad(x_padded, num_branches: int):
            # the values that we unpad have twice the batch size
            # since they contain first the left branch and
            # then the right branches
            left = x_padded[:num_branches]
            right = x_padded[batch_size : batch_size + num_branches]
            return jnp.concat([left, right], axis=0)

        def bab_step(batch: BranchData, num_branches: int):
            padding = batch_size - num_branches
            padded = jax.tree.map(partial(pad, padding=padding), batch)
            out_padded = bab_step_jit(padded, num_branches=batch_size)
            out = jax.tree.map(partial(unpad, num_branches=num_branches), out_padded)
            return out

    # Root branch
    all_coalitions = Box(
        jnp.zeros((1,) + base_mask.shape, dtype=base_mask.dtype),
        jnp.ones((1,) + base_mask.shape, dtype=base_mask.dtype),
    )
    zero = jnp.zeros((1,), dtype=int)
    total_coaliw = jnp.ones((1,), dtype=base_mask.dtype)
    value_lb, value_ub = bound_value(all_coalitions).concrete
    root_data = BranchData(all_coalitions, value_lb, value_ub, zero, total_coaliw, zero)

    branches: PriorityBranchStore[BranchData] = PriorityBranchStore(
        selection_priority(root_data), root_data, batch_size
    )

    shap_lbs, shap_ubs = jax.tree.map(lambda x: x.squeeze(), shap_bounds(root_data))
    total_tight_bounds = total_fully_split = 0
    yield Box(shap_lbs.squeeze(), shap_ubs.squeeze())
    for i in it.count():
        if len(branches) == 0 or jnp.allclose(shap_lbs, shap_ubs):
            break

        with timer["extract_max"]:
            batch, num_selected = branches.extract_max(return_size=True)
        with timer["bab_step"]:
            (
                new_branches,
                ((old_shap_lbs, old_shap_ubs), (new_shap_lbs, new_shap_ubs)),
                (prune, single_coalition, tight_bounds),
            ) = bab_step(batch, num_selected)
        with timer["update_shap_bounds"]:
            # the shap_lbs and shap_ubs still have branch axes
            shap_lbs = shap_lbs - old_shap_lbs.sum(axis=0) + new_shap_lbs.sum(axis=0)
            shap_ubs = shap_ubs - old_shap_ubs.sum(axis=0) + new_shap_ubs.sum(axis=0)
        yield Box(shap_lbs.squeeze(), shap_ubs.squeeze())

        with timer["drop_pruned"]:
            # shapley bounds from the pruned branches remain part of the global bounds
            pruned = jax.tree.map(partial(drop_pruned, prune), new_branches)

        with timer["insert_branches"]:
            branches.insert(selection_priority(pruned), pruned)

        num_fully_split = single_coalition.sum().item()
        num_tight = (tight_bounds & ~single_coalition).sum().item()
        total_fully_split += num_fully_split
        total_tight_bounds += num_tight

        if log is not False:
            num_branches = len(branches)
            log.log_iter_stats(
                "multi_shap_bab",
                i,
                num_branches=num_branches,
                num_fully_split=num_fully_split,
                num_tight=num_tight,
            )
            log.log_bounds("multi_shap_bab", i, (shap_lbs, shap_ubs), name="φ")

    total_branches = total_tight_bounds + total_fully_split
    log.log_stats(
        "multi_shap_bab",
        {"runtimes": timer.runtimes, "iterations": i + 1},
        total_branches=total_branches,
        total_tight_bounds=total_tight_bounds,
        total_fully_split=total_fully_split,
    )
