#  Copyright (c) 2025. David Boetius.
from typing import Callable

from jaxtyping import Array, Real

from ..leverageshap import ModelWrapper
from ..shaplib import _preprocess_array
from .regMSR import LinearMSR, TreeMSR


def reg_msr(
    model: Callable[[Real[Array, " b *n"]], Real[Array, " b m"]],
    baseline: Real[Array, " d *n"],
    x: Real[Array, " *n"],
    explainer_class,
    num_samples: int | None = None,
    batch_size: int = 16384,
):
    if isinstance(baseline, Callable):
        raise ValueError("KernelSHAP does not support callable maskers.")

    model = ModelWrapper(model, batch_size, squeeze=True)
    x = _preprocess_array(x)
    baseline = _preprocess_array(baseline)

    explainer = explainer_class(model, baseline, weighting="shapley")
    return explainer.explain(x, num_samples).reshape(-1, 1)


def linear_msr(
    model: Callable[[Real[Array, " b *n"]], Real[Array, " b m"]],
    baseline: Real[Array, " d *n"],
    x: Real[Array, " *n"],
    num_samples: int | None = None,
    batch_size: int = 16384,
):
    return reg_msr(model, baseline, x, LinearMSR, num_samples, batch_size)


def tree_msr(
    model: Callable[[Real[Array, " b *n"]], Real[Array, " b m"]],
    baseline: Real[Array, " d *n"],
    x: Real[Array, " *n"],
    num_samples: int | None = None,
    batch_size: int = 16384,
):
    return reg_msr(model, baseline, x, TreeMSR, num_samples, batch_size)
