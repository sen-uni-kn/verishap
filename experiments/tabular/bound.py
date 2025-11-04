# Copyright 2025 David Boetius
import argparse
import itertools as it
from datetime import datetime, timezone
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from shap_bounds import baseline_value, marginal_value, shapley_bab

from .models import MLP
from .utils import load_dataset

local_resoure_dir = Path(__file__).parent / "resources"
local_output_dir = Path(__file__).parent / "output"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Extracted from model file name if not provided.",
    )
    parser.add_argument(
        "--input",
        type=int,
        default=0,
        help="The index of the input sample to analyse in the dataset.",
    )
    parser.add_argument(
        "--feature",
        type=int,
        default=0,
        help="The index of the input feature to compute SHAP bounds for.",
    )
    parser.add_argument(
        "--output-feature",
        type=int,
        default=0,
        help="The index of the output feature to explain.",
    )
    parser.add_argument(
        "--shap-variant",
        type=str,
        default="zero-baseline",
        help="The SHAP variant to use. Options: zero-baseline, marginal-shap",
    )
    parser.add_argument(
        "--bound-method",
        type=str,
        default="bab",
        help="The method to use for computing the bounds.",
    )
    parser.add_argument(
        "--bound-options",
        type=str,
        default="",
        help="Keyword arguments to pass to the bound method in the format "
        "key1=value1,key2=value2,...",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=None,
        help="How many iterations to perform at most.",
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default=None,
        help="Where to save the experiment results.",
    )
    args = parser.parse_args()

    np.random.seed(0)

    dataset = args.dataset
    if dataset is None:
        dataset = args.model.stem.split("-")[0]
    data, targets = load_dataset(dataset)

    model = MLP.load(args.model)
    model = eqx.nn.inference_mode(model)
    model = jax.vmap(model, axis_name="batch")

    in_feature = args.feature
    out_feature = args.output_feature
    x = data[args.input]

    match args.shap_variant:
        case "zero-baseline":
            baseline = jnp.zeros_like(x)
            value_fn = baseline_value(model, baseline, out_feature)
        case "marginal-shap":
            num_background = 100
            rng = np.random.default_rng(0)
            perm = rng.permutation(len(data))
            background = data[perm[:num_background]]
            value_fn = marginal_value(model, background, out_feature)
        case _:
            raise ValueError(f"Unknown SHAP variant: {args.shap_variant}")

    match args.bound_method:
        case "bab":
            bounds_method = shapley_bab
        case _:
            raise ValueError(f"Unknown bound method: {args.bound_method}")

    bound_kwargs = {}
    if args.bound_options:
        for option in args.bound_options.split(","):
            k, v = option.split("=")
            try:
                v = eval(v, {})
            except NameError:
                pass
            bound_kwargs[k] = v

    bounds_iter = bounds_method(value_fn, x, in_feature, **bound_kwargs)
    iters = range(args.max_iters) if args.max_iters is not None else it.count()
    bounds = []
    for (lb, ub), _ in zip(bounds_iter, iters, strict=False):
        bounds.append((lb, ub))

    print("Best Bound: ", bounds[-1][0], bounds[-1][1])

    if args.out_file is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        out_file = (
            local_output_dir
            / f"{dataset}"
            / f"{args.model.stem}_{in_feature}_{out_feature}_shap_{args.bound_method}_{args.shap_variant}_{timestamp}.csv"
        )
    else:
        out_file = Path(args.out_file)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    bounds = pd.DataFrame(bounds, columns=["lb", "ub"])
    bounds.to_csv(out_file, index=False)
