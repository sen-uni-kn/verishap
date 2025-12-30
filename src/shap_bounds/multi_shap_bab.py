# Copyright 2025 David Boetius
import itertools as it
from collections.abc import Sequence
from dataclasses import astuple, dataclass
from math import prod
from time import perf_counter
from typing import Callable, Literal
from warnings import warn

import jax
import jax.numpy as jnp
import numpy as np
from formalax import Box, alpha_crown, crown, crown_ibp, ibp
from jaxtyping import Array, Bool, Int, Real

from .branch_queue import BranchQueue
from .branch_stack import BranchStack
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
    pruned: Bool[Array, " b"]  # whether the branch has been pruned


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


def resolve_compute_bounds(
    compute_bounds: Callable | Literal["crown_ibp", "ibp", "crown", "alpha-crown"],
):
    if isinstance(compute_bounds, str):
        match compute_bounds:
            case "crown_ibp":
                return crown_ibp
            case "ibp":
                return ibp
            case "crown":
                return crown
            case "alpha-crown" | "alpha_crown":
                return alpha_crown
            case _:
                raise ValueError(f"Unknown compute bounds method: {compute_bounds}")
    return compute_bounds


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
    ] = "smears",
    batch_size: int = 1024,
    storage_batch_size: int = 256,
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

    compute_bounds = resolve_compute_bounds(compute_bounds)
    fast_compute_bounds = resolve_compute_bounds(fast_compute_bounds)

    if features is None:
        features = jnp.indices(base_mask.shape).reshape(base_mask.ndim, -1).T
        features = [tuple(f) for f in features.tolist()]
    elif isinstance(features, int):
        features = [features]

    max_num_branches = 2 ** prod(base_mask.shape)
    if batch_size > max_num_branches:
        batch_size = max_num_branches

    if log is True:
        log = ConsoleLogger()

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
    start_time = perf_counter()

    data_axes = tuple(range(1, base_mask.ndim + 1))
    bound_value = compute_bounds(value_fn)
    fast_bound_value = fast_compute_bounds(value_fn)
    value_smears = smears(value_fn)

    def selection_priority(branches: BranchData):
        """Computes the priority of branches that determine the order of selection."""
        match select_strategy:
            case "max-diam" | "min-diam":
                total_coaliw = branches.total_coalition_weight
                diameters = (branches.value_ub - branches.value_lb) * total_coaliw
                scores = diameters if select_strategy == "max-diam" else -diameters
                return jnp.where(branches.pruned, -jnp.inf, scores)
            case "fifo" | "lifo":
                return None
            case _:
                raise ValueError(f"Invalid select strategy: {select_strategy}")

    def split(
        coalitions: Box[Real[Array, " b *shape"]],
        num_splits: Int[Array, " b"],
        total_coalition_weight: Real[Array, " b"],
        split_scores: Real[Array, " b f"],
    ) -> tuple[Box[Real[Array, " b *shape"]], Real[Array, " b"]]:
        """Split branches by including/excluding one feature.

        Args:
            branches: The branches to split.
            split_scores: Scores indicating which feature to split.
                The feature with the highest score is split.

        Returns:
            The coalition bounds and the total coalition weights after splitting.
        """
        coali_lb, coali_ub = coalitions
        num_branches = coali_lb.shape[0]
        coali_lb_ = jnp.reshape(coali_lb, (num_branches, -1))
        coali_ub_ = jnp.reshape(coali_ub, (num_branches, -1))

        # mask out fully split branches
        split_scores = jnp.where(coali_lb_ != coali_ub_, split_scores, -jnp.inf)
        split_axes = jnp.argmax(split_scores, axis=-1)

        left_ub = coali_ub_.at[np.arange(num_branches), split_axes].set(0.0)
        right_lb = coali_lb_.at[np.arange(num_branches), split_axes].set(1.0)
        left_ub = jnp.reshape(left_ub, coali_lb.shape)
        right_lb = jnp.reshape(right_lb, coali_lb.shape)

        s, r = num_splits, coali_lb_.sum(axis=-1)
        old_total_coaliw = total_coalition_weight
        # left branch is exclude branch
        left_total_coaliw = (s + 1 - r) / (s + 2) * old_total_coaliw
        right_total_coaliw = (r + 1) / (s + 2) * old_total_coaliw

        new_coali_lb = jnp.concat([coali_lb, right_lb])
        new_coali_ub = jnp.concat([left_ub, coali_ub])
        new_total_coaliw = jnp.concat([left_total_coaliw, right_total_coaliw])
        return Box(new_coali_lb, new_coali_ub), new_total_coaliw

    def select_split(branches: BranchData):
        """Computes split scores for each feature of each branch."""
        coali_lb, coali_ub = branches.coalitions

        num_branches = coali_lb.shape[0]
        coali_lb_ = jnp.reshape(coali_lb, (num_branches, -1))
        coali_ub_ = jnp.reshape(coali_ub, (num_branches, -1))

        if split_strategy == "longest-edge":
            edge_len = jnp.abs(coali_ub_ - coali_lb_)
            return edge_len
        elif split_strategy == "smears":
            grad_lbs, grad_ubs = value_smears(branches.coalitions).concrete
            smears_ = jnp.maximum(jnp.abs(grad_lbs), jnp.abs(grad_ubs))
            return smears_
        elif split_strategy == "lirpa-weights":
            lirpa_bounds = bound_value(branches.coalitions)
            lb_weights = lirpa_bounds.lb_weights[0]
            ub_weights = lirpa_bounds.ub_weights[0]
            return jnp.maximum(jnp.abs(lb_weights), jnp.abs(ub_weights))
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
                select_i = (
                    jnp.zeros(
                        (
                            1,
                            num_features,
                        ),
                    )
                    .at[i]
                    .set(1.0)
                )
                select_i = jnp.repeat(select_i, num_branches, axis=0)
                split_coalis, _ = split(
                    branches.coalitions,
                    branches.num_splits,
                    branches.total_coalition_weight,
                    select_i,
                )
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
                return carry, score

            _, split_scores = jax.lax.scan(eval_split, None, jnp.arange(num_features))
            return -jnp.moveaxis(split_scores, 0, -1)
        else:
            raise ValueError(f"Invalid split strategy: {split_strategy}")

    def to_prune(coalitions, value_lb, value_ub) -> tuple[Bool[Array, " b"], ...]:
        """Determine which branches to prune."""
        # Prune completely split branches
        coali_lb, coali_ub = coalitions
        single_coalition = (coali_lb == coali_ub).all(axis=data_axes)
        # Also prune branches with tight value bounds
        tight_bounds = jnp.isclose(value_ub, value_lb) | jnp.isclose(value_lb, value_ub)
        invalid_bounds = ~tight_bounds & (value_ub < value_lb)
        prune = single_coalition | tight_bounds | invalid_bounds
        return prune, single_coalition, tight_bounds, invalid_bounds

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

    def shap_bounds(
        coalitions, value_lb, value_ub, num_splits, total_coalition_weight, pruned
    ) -> Box:
        coali_lb, coali_ub = coalitions

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
        s, r = num_splits, coali_lb.sum(axis=data_axes)
        sum_coaliw = total_coalition_weight
        # Reduce number of included features by one
        sum_coaliw_with_i = (s + 1) / r * sum_coaliw
        # Reduce number of excluded features by one
        sum_coaliw_without_i = (s + 1) / (s - r) * sum_coaliw

        # Deal with invalid value bounds (originating from floating point errors)
        value_mid = (value_lb + value_ub) / 2
        value_lb = jnp.where(value_ub < value_lb, value_mid, value_lb)
        value_ub = jnp.where(value_ub < value_lb, value_mid, value_ub)

        # Computing bounds on: sum lambda * v(S + {i}) - sum lambda * v(S)
        shap_lb: Real[Array, " f b"] = (
            sum_coaliw_with_i * value_lb * with_i
            + sum_coaliw * value_lb * both
            - sum_coaliw * value_ub * both
            - sum_coaliw_without_i * value_lb * without_i
        )
        shap_ub = (
            sum_coaliw_with_i * value_ub * with_i
            + sum_coaliw * value_ub * both
            - sum_coaliw * value_lb * both
            - sum_coaliw_without_i * value_lb * without_i
        )
        shap_lb = jnp.where(pruned, 0.0, shap_lb).sum(axis=-1)
        shap_ub = jnp.where(pruned, 0.0, shap_ub).sum(axis=-1)
        return Box(shap_lb, shap_ub)

    def bab_step(
        batch: BranchData,
    ) -> tuple[
        BranchData,
        Real[Array, ""],
        tuple[Box, Box],
        Sequence[Bool[Array, " b"]],
    ]:
        old_shap_bounds = shap_bounds(*astuple(batch))

        split_scores = select_split(batch)
        new_coalitions, new_total_coaliw = split(
            batch.coalitions,
            batch.num_splits,
            batch.total_coalition_weight,
            split_scores,
        )
        new_value_lbs, new_value_ubs = bound_value(new_coalitions).concrete
        new_num_splits = jnp.concat(
            [batch.num_splits + 1, batch.num_splits + 1], axis=0
        )

        # Need to compute new shap bounds with old pruning status
        new_shap_bounds = shap_bounds(
            new_coalitions,
            new_value_lbs,
            new_value_ubs,
            new_num_splits,
            new_total_coaliw,
            pruned=jnp.concat([batch.pruned, batch.pruned], axis=0),
        )

        prune, *prune_log_info = to_prune(new_coalitions, new_value_lbs, new_value_ubs)
        new_branches = BranchData(
            new_coalitions,
            new_value_lbs,
            new_value_ubs,
            new_num_splits,
            new_total_coaliw,
            prune,
        )
        priority = selection_priority(new_branches)
        return (
            new_branches,
            priority,
            (old_shap_bounds, new_shap_bounds),
            prune_log_info,
        )

    def drop_pruned(branches: BranchData) -> BranchData:
        prune = branches.pruned
        return jax.tree.map(lambda a: a[~prune], branches)

    if jit:
        bab_step = jax.jit(bab_step)

    # Root branch
    all_coalitions = Box(
        jnp.zeros((1,) + base_mask.shape, dtype=base_mask.dtype),
        jnp.ones((1,) + base_mask.shape, dtype=base_mask.dtype),
    )
    value_lb, value_ub = bound_value(all_coalitions).concrete
    root_data = BranchData(
        all_coalitions,
        value_lb,
        value_ub,
        num_splits=jnp.zeros((1,), dtype=int),
        total_coalition_weight=jnp.ones((1,), dtype=base_mask.dtype),
        pruned=jnp.full((1,), False),
    )

    # Prefill one batch of branches without computing bounds
    split_scores = select_split(root_data)
    coalitions = root_data.coalitions
    num_splits = root_data.num_splits
    total_coaliw = root_data.total_coalition_weight
    while coalitions.shape[0] < batch_size:
        coalitions, total_coaliw = split(
            coalitions, num_splits, total_coaliw, split_scores
        )
        num_splits = jnp.concat([num_splits + 1, num_splits + 1], axis=0)

    value_lb, value_ub = bound_value(coalitions).concrete
    prune, single_coalition, tight_bounds, invalid_bounds = to_prune(
        coalitions, value_lb, value_ub
    )
    # Since we limited the batch size to be at maximum the total number of possible
    # branches, the branches may be fully split, but not split in an invalid way
    # so far.
    shap_lbs, shap_ubs = shap_bounds(
        coalitions,
        value_lb,
        value_ub,
        num_splits,
        total_coaliw,
        jnp.full((coalitions.shape[0],), False),
    )
    prefilled = BranchData(
        coalitions, value_lb, value_ub, num_splits, total_coaliw, prune
    )

    total_fully_split = num_fully_split = single_coalition.sum().item()
    total_tight_bounds = num_tight = (tight_bounds & ~single_coalition).sum().item()
    total_invalid_bounds = num_invalid_bounds = invalid_bounds.sum().item()
    total_pruned = prune.sum().item()

    if log is not False:
        num_branches = coalitions.shape[0]
        log.log_iter_stats(
            "multi_shap_bab",
            0,
            num_branches=num_branches,
            num_fully_split=num_fully_split,
            num_tight=num_tight,
            num_invalid_bounds=num_invalid_bounds,
        )
        log.log_bounds(
            "multi_shap_bab",
            0,
            Box(shap_lbs, shap_ubs),
            name="φ",
            runtime=perf_counter() - start_time,
        )

    yield Box(shap_lbs.squeeze(), shap_ubs.squeeze())

    match select_strategy:
        case "fifo":
            pruned = drop_pruned(prefilled)
            branches = BranchQueue(pruned, batch_size)
        case "lifo":
            pruned = drop_pruned(prefilled)
            branches = BranchStack(pruned, batch_size)
        case _:
            branches = PriorityBranchStore(
                selection_priority(prefilled), prefilled, batch_size
            )
            branches.drop(max_priority=-float("inf"))

    for i in it.count(1):
        if (
            len(branches) == 0
            or jnp.allclose(shap_lbs, shap_ubs)
            or jnp.allclose(shap_ubs, shap_lbs)  # allclose is not symmetric
        ):
            break

        batch = branches.pop()
        (
            new_branches,
            priority,
            ((old_shap_lbs, old_shap_ubs), (new_shap_lbs, new_shap_ubs)),
            (single_coalition, tight_bounds, invalid_bounds),
        ) = bab_step(batch)

        shap_lbs = shap_lbs - old_shap_lbs + new_shap_lbs
        shap_ubs = shap_ubs - old_shap_ubs + new_shap_ubs
        yield Box(shap_lbs.squeeze(), shap_ubs.squeeze())

        if isinstance(branches, PriorityBranchStore):
            branches.insert(priority, new_branches)
            # drops pruned branches if they fill a batch
            branches.drop(max_priority=-float("inf"))
        else:
            # shap bounds from the pruned branches remain part of the global bounds
            pruned = drop_pruned(new_branches)
            branches.insert(pruned)

        num_fully_split = single_coalition.sum().item()
        num_tight = (tight_bounds & ~single_coalition).sum().item()
        num_invalid_bounds = invalid_bounds.sum().item()
        num_pruned = (single_coalition | tight_bounds | invalid_bounds).sum().item()

        if num_invalid_bounds > 0 and total_invalid_bounds == 0:
            warn(
                "Encountered invalid value bounds. "
                f"Number of branches with invalid bounds: {num_invalid_bounds}",
                stacklevel=1,
            )

        total_invalid_bounds += num_invalid_bounds
        total_fully_split += num_fully_split
        total_tight_bounds += num_tight
        total_pruned += num_pruned

        if log is not False:
            num_branches = len(branches)
            log.log_iter_stats(
                "multi_shap_bab",
                i,
                num_branches=num_branches,
                num_fully_split=num_fully_split,
                num_tight=num_tight,
                num_invalid_bounds=num_invalid_bounds,
            )
            log.log_bounds(
                "multi_shap_bab",
                i,
                Box(shap_lbs, shap_ubs),
                name="φ",
                runtime=perf_counter() - start_time,
            )
            total_branches = total_tight_bounds + total_fully_split
            log.log_stats(
                "multi_shap_bab",
                {"iterations": i},
                temporary=True,
                total_branches=total_branches,
                total_tight_bounds=total_tight_bounds,
                total_fully_split=total_fully_split,
                total_invalid_bounds=total_invalid_bounds,
            )

    total_branches = total_pruned
    if log is not False:
        log.log_stats(
            "multi_shap_bab",
            {"iterations": i},
            total_branches=total_branches,
            total_tight_bounds=total_tight_bounds,
            total_fully_split=total_fully_split,
            total_invalid_bounds=total_invalid_bounds,
        )
