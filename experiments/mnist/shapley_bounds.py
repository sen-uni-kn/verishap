# Copyright 2025 David Boetius
import argparse
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import pandas as pd
import torchvision

from shap_bounds import baseline_value, shapley_bab

from .models import CNN

local_resoure_dir = Path(__file__).parent / "resources"
local_output_dir = Path(__file__).parent / "output"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default=local_resoure_dir / "mnist-cnn.eqxparams",
        help="The path of the model file to load.",
    )
    parser.add_argument(
        "--input", type=int, default=0,
        help="The index of the input image to analyse in the MNIST test set.",
    )
    parser.add_argument(
        "--feature", type=tuple[int, int], default=(0, 0),
        help="The input feature to compute SHAP bounds for.",
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

    model = eqx.tree_deserialise_leaves(args.model, CNN(jax.random.PRNGKey(0)))
    model = jax.vmap(model)

    testset = torchvision.datasets.MNIST(
        ".datasets",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    x = testset.data[args.input].numpy()
    x = x.reshape(1, 28, 28) / 255.0

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
