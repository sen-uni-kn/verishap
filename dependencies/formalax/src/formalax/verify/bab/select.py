#  Copyright (c) 2025. The Formalax Authors.
#  Licensed under the MIT license.
import jax
from jaxtyping import Array, Real

from ...bounds._src._bounds import Bounds
from .bab import (
    BAB_TARGET,
    ArrayRef,
    BranchSelection,
    BranchStore,
    PyTree,
    SelectBranches,
)

__all__ = [
    "select_worst",
    "select_best",
]


def select_worst[BS: BranchSelection](
    jaxpr: jax.extend.core.ClosedJaxpr,
    *args: Bounds | Real[Array, "..."],
    target: BAB_TARGET,
) -> SelectBranches[ArrayRef[BranchStore, BS], BS]:
    """Selects the branches with the currently worst output bounds.

    If the target is minimization, the branches with the smallest
    lower bounds are selected.
    For maximization, the branches with the largest upper bounds are
    selected.
    For the function image (``target="img"``), the branches with
    the largest difference between upper and lower bound are selected.

    This is a factory function that creates a select procedure.
    """

    def min_worst(branches, out_lbs, out_ubs, select_size):
        return out_lbs.argmin_k(select_size)

    def max_worst(branches, out_lbs, out_ubs, select_size):
        return out_ubs.argmax_k(select_size)

    def img_worst(branches, out_lbs, out_ubs, select_size):
        return (out_ubs - out_lbs).argmax_k(select_size)

    worst = {
        "min": min_worst,
        "max": max_worst,
        "img": img_worst,
    }[target]

    def select(
        branches: PyTree[ArrayRef[BranchStore, BS], "..."],
        out_lbs: ArrayRef[BranchStore, BS],
        out_ubs: ArrayRef[BranchStore, BS],
        batch_size: int,
    ) -> BS:
        select_size = min(batch_size, len(out_lbs))
        return worst(branches, out_lbs, out_ubs, select_size)

    return select


def select_best[BS: BranchSelection](
    jaxpr: jax.extend.core.ClosedJaxpr,
    *args: Bounds | Real[Array, "..."],
    target: BAB_TARGET,
) -> SelectBranches[ArrayRef[BranchStore, BS], BS]:
    """Selects the branches with the currently best output bounds.

    If the target is minimization, the branches with the largest
    lower bounds are selected.
    For maximization, the branches with the smallest upper bounds are
    selected.
    For the function image (``target="img"``), the branches with
    the smallest difference between upper and lower bound are selected.

    This is a factory function that creates a select procedure.
    """

    def min_best(branches, out_lbs, out_ubs, select_size):
        return out_lbs.argmax_k(select_size)

    def max_best(branches, out_lbs, out_ubs, select_size):
        return out_ubs.argmin_k(select_size)

    def img_best(branches, out_lbs, out_ubs, select_size):
        return (out_ubs - out_lbs).argmin_k(select_size)

    best = {
        "min": min_best,
        "max": max_best,
        "img": img_best,
    }[target]

    def select(
        branches: PyTree[ArrayRef[BranchStore, BS], "..."],
        out_lbs: ArrayRef[BranchStore, BS],
        out_ubs: ArrayRef[BranchStore, BS],
        batch_size: int,
    ) -> BS:
        select_size = min(batch_size, len(out_lbs))
        return best(branches, out_lbs, out_ubs, select_size)

    return select
