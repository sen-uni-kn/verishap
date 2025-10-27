# Copyright 2025 David Boetius
import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd

from shap_bounds import baseline_value, shapley_bab


local_resoure_dir = Path(__file__).parent / "resources"
local_output_dir = Path(__file__).parent / "output"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=tuple[int, int], default=(2, 1),
        help="The indices of the ACAS Xu model to load.",
    )
    parser.add_argument(
        "--feature", type=int, default=0,
        help="The index of the input feature to compute SHAP bounds for.",
    )
    parser.add_argument(
        "--output-feature", type=int, default=0,
        help="The index of the output feature to explain.",
    )
    parser.add_argument(
        "--shap-variant", type=str, default="zero-baseline",
        help="The SHAP variant to use.",
    )
    parser.add_argument(
        "--bound-method", type=str, default="bab",
        help="The method to use for computing the bounds.",
    )
    parser.add_argument(
        "--out-file", type=str, default=local_output_dir / "shapley_bounds.csv",
        help="Where to save the experiment results.",
    )
    args = parser.parse_args()

    np.random.seed(0)

    model = acasxu_network(*args.model)

    x = np.random.uniform(model.in_min, model.in_max, model.in_means.shape)

    in_feature = args.feature
    out_feature = args.output_feature

    match args.shap_variant:
        case "zero-baseline":
            baseline = jnp.zeros_like(x)
            value_fn = baseline_value(model, baseline, out_feature)
        case _:
            raise ValueError(f"Unknown SHAP variant: {args.shap_variant}")

    match args.bound_method:
        case "bab":
            bounds_method = shapley_bab
        case _:
            raise ValueError(f"Unknown bound method: {args.bound_method}")

    bounds = []
    for lb, ub in bounds_method(value_fn, x, in_feature):
        print(lb, ub)
        bounds.append((lb, ub))

    bounds = pd.Series(bounds)
    bounds.to_csv(args.output, index=False)
