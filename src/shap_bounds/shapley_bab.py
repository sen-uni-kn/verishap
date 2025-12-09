# Copyright 2025 David Boetius
import itertools as it
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np
from formalax import Box, crown_ibp, ibp
from jaxtyping import Array, Int, Real

from .priority_branch_store import PriorityBranchStore


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


def shapley_bab(
    value_fn: Callable[[Real[Array, " b *shape"]], Real[Array, " b"]],
    base_mask: Real[Array, " *shape"],
    features: Sequence[tuple[int, ...]] | None = None,
    compute_bounds=crown_ibp,
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
    if features is None:
        features = jnp.indices(base_mask.shape).reshape(base_mask.ndim, -1).T
        features = [tuple(f) for f in features.tolist()]
    data_axes = tuple(range(1, base_mask.ndim + 1))

    bound_value = compute_bounds(value_fn)
    fast_bound_value = fast_compute_bounds(value_fn)
    value_smears = smears(value_fn)

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

        num_splits = branches.num_splits
        # The coalition weights assume one feature is excluded for
        # which the Shapley value is computed.
        # However, num_splits is the overall number of splits starting
        # from the entire feature space.
        # This is one larger than the proper value of `s` for computing
        # the coalition weights.
        # For the formula below to work out, we set the initial total_coaliw
        # for the entire feature space to 2.
        # With taking s = min(num_splits - 1, 0), that produces the
        # right coalition weights for num_splits=2.
        # Same for the number of included features (r).
        s = jnp.maximum(num_splits - 1, jnp.zeros_like(num_splits))
        r = jnp.maximum(coali_lb_.sum(axis=-1), s)
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

    def bab_step(batch: BranchData, num_branches: int):
        split_axes = select_split(batch)
        new_coalitions, new_total_coaliw = split(batch, split_axes)
        new_value_lbs, new_value_ubs = bound_value(new_coalitions).concrete
        new_num_splits = jnp.concat(
            [batch.num_splits + 1, batch.num_splits + 1], axis=0
        )
        return BranchData(
            new_coalitions,
            new_value_lbs,
            new_value_ubs,
            new_num_splits,
            new_total_coaliw,
        )

    def to_prune(branches: BranchData, num_branches: int):
        """These steps can be awefully slow for some reason if not jitted."""
        # Prune completely split branches
        coali_lb, coali_ub = branches.coalitions
        single_coalition = (coali_lb == coali_ub).all(axis=data_axes)
        # Also prune branches with tight value bounds
        tight_bounds = jnp.isclose(branches.value_ub, branches.value_lb)
        prune = single_coalition | tight_bounds
        return prune, single_coalition, tight_bounds

    if jit:
        bab_step_jit = jax.jit(bab_step, static_argnums=(1,))
        to_prune_jit = jax.jit(to_prune, static_argnums=(1,))

        def pad(x, padding: int):
            pad_val = jnp.empty((padding, *x.shape[1:]), dtype=x.dtype)
            return jnp.concat([x, pad_val], axis=0)

        def bab_step(batch: BranchData, num_branches: int):
            padding = batch_size - num_branches

            def unpad(x_padded):
                # the values that we unpad have twice the batch size
                # since they contain first the left branch and
                # then the right branches
                left = x_padded[:num_branches]
                right = x_padded[batch_size : batch_size + num_branches]
                return jnp.concat([left, right], axis=0)

            padded = jax.tree.map(partial(pad, padding=padding), batch)
            out_padded = bab_step_jit(padded, num_branches=batch_size)
            out = jax.tree.map(unpad, out_padded)
            return out

        def to_prune(branches: BranchData, num_branches: int):
            padding = 2 * batch_size - num_branches

            def unpad(x_padded):
                return x_padded[:num_branches]

            padded = jax.tree.map(partial(pad, padding=padding), branches)
            res = to_prune_jit(padded, num_branches=batch_size)
            return jax.tree.map(unpad, res)

    branch_counter = it.count()

    def selection_priority(branches: BranchData):
        """Computes the priority of branches that determine the order of selection."""
        match select_strategy:
            case "fifo" | "lifo":
                scores = jnp.array([next(branch_counter) for _ in range(len(branches))])
                return scores if select_strategy == "lifo" else -scores
            case "max-diam" | "min-diam":
                total_coaliw = branches.total_coalition_weight
                diameters = (branches.value_ub - branches.value_lb) * total_coaliw
                return diameters if select_strategy == "max-diam" else -diameters
            case _:
                raise ValueError(f"Invalid select strategy: {select_strategy}")

    # Root branch
    all_coalitions = Box(
        jnp.zeros((1,) + base_mask.shape, dtype=base_mask.dtype),
        jnp.ones((1,) + base_mask.shape, dtype=base_mask.dtype),
    )
    zero_depth = jnp.zeros((1,), dtype=int)
    # refined to 1.0 for all branches in the first iteration
    total_coaliw = 2 * jnp.ones((1,), dtype=base_mask.dtype)
    value_lb, value_ub = bound_value(all_coalitions).concrete
    root_data = BranchData(all_coalitions, value_lb, value_ub, zero_depth, total_coaliw)

    branches: PriorityBranchStore[BranchData] = PriorityBranchStore(
        selection_priority(root_data), root_data, batch_size
    )

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

        # feature not necessarily excluded
        not_excluded: Real[Array, " f b"]
        not_excluded = (coali_ub >= with_features).all(axis=data_axes2)
        # feature not necessarily included
        not_included = (coali_lb <= without_features).all(axis=data_axes2)

        wv_lb = val_lb * branches.total_coalition_weight
        wv_ub = val_ub * branches.total_coalition_weight
        contrib_lb = (val_lb * not_excluded - val_ub * not_included).sum(axis=-1)
        contrib_ub = (wv_ub * not_excluded - wv_lb * not_included).sum(axis=-1)
        return Box(contrib_lb, contrib_ub)

    shap_lbs, shap_ubs = shap_bounds(root_data)
    yield Box(shap_lbs.squeeze(), shap_ubs.squeeze())
    for i in it.count():
        if len(branches) == 0 or jnp.allclose(shap_lbs, shap_ubs):
            return None

        batch, num_selected = branches.extract_max(return_size=True)

        print("Before", batch.total_coalition_weight)

        # Subtract here, since we will refine the bounds for batch
        batch_shap_lbs, batch_shap_ubs = shap_bounds(batch)
        shap_lbs = shap_lbs - batch_shap_lbs
        shap_ubs = shap_ubs - batch_shap_ubs

        new_branches = bab_step(batch, num_selected)

        print("After", new_branches.total_coalition_weight)

        new_shap_lbs, new_shap_ubs = shap_bounds(new_branches)
        shap_lbs = shap_lbs + new_shap_lbs
        shap_ubs = shap_ubs + new_shap_ubs
        yield Box(shap_lbs.squeeze(), shap_ubs.squeeze())

        # shapley bounds from the pruned branches remain part of the global bounds
        num_branches = new_branches.value_lb.shape[0]
        prune, single_coalition, tight_bounds = to_prune(new_branches, num_branches)
        pruned = jax.tree.map(lambda a: a[~prune], new_branches)  # noqa: B023
        branches.insert(selection_priority(pruned), pruned)

        if log:
            num_branches = len(branches)
            num_fully_split = single_coalition.sum()
            num_tight = (tight_bounds & ~single_coalition).sum()
            lbs, ubs = shap_lbs, shap_ubs
            mid, ran = (lbs + ubs) / 2, (ubs - lbs) / 2
            mid, ran = mid.tolist(), ran.tolist()
            print(f"[i: {i:3d}] Branches: {num_branches}, Pruned: {num_tight} tight, {num_fully_split} fully split")
            bounds = ", ".join(
                [f"{m:.4f} ± {r:.4f}" for m, r in zip(mid, ran, strict=True)]
            )
            print(f"    φ ∈ [{bounds}]")
