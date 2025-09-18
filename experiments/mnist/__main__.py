# Copyright 2025 David Boetius
import argparse
import itertools as it
from datetime import datetime, timezone
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import pandas as pd
import torchvision

from shap_bounds import shapley_bab, superfeature_baseline_value

from .models import CNN

local_resoure_dir = Path(__file__).parent / "resources"
local_output_dir = Path(__file__).parent / "output"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=local_resoure_dir / "mnist-cnn.eqxparams",
        help="The path of the model file to load.",
    )
    parser.add_argument(
        "--input",
        type=int,
        default=0,
        help="The index of the input image to analyse in the MNIST test set.",
    )
    parser.add_argument(
        "--num-patches",
        type=str,
        default="28,28",
        help="The number of patches to consider as features.",
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="0",
        help="The input feature to compute SHAP bounds for.",
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
        help="The SHAP variant to use.",
    )
    parser.add_argument(
        "--bound-method",
        type=str,
        default="bab",
        help="The method to use for computing the bounds.",
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

    num_patches = args.num_patches
    try:
        num_patches = int(num_patches)
        num_patches = (num_patches, num_patches)
    except ValueError:
        num_patches = tuple(int(x) for x in num_patches.split(","))
    assert num_patches[0] <= 28
    assert num_patches[1] <= 28
    total_patches = num_patches[0] * num_patches[1]
    num_patches = (1, *num_patches)

    in_feature = args.feature
    try:
        in_feature = int(in_feature)
    except ValueError:
        in_feature = tuple(int(x) for x in in_feature.split(","))
        in_feature = in_feature[0] * num_patches[0] + in_feature[1]
    out_feature = args.output_feature

    model = eqx.tree_deserialise_leaves(args.model, CNN(jax.random.PRNGKey(0)))
    model = jax.vmap(model)

    # create patch masks
    img_size = (1, 28, 28)
    mask_idx = jnp.arange(total_patches).reshape(num_patches)
    mask_idx = jax.image.resize(mask_idx, img_size, "nearest")
    masks = jnp.arange(total_patches).reshape((total_patches, 1, 1, 1)) == mask_idx

    testset = torchvision.datasets.MNIST(
        ".datasets",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    sample = testset.data[args.input].numpy()
    sample = sample.reshape(1, 28, 28) / 255.0

    match args.shap_variant:
        case "zero-baseline":
            baseline = jnp.zeros_like(sample)
            value_fn = superfeature_baseline_value(
                model, sample, baseline, masks, out_feature
            )
        case _:
            raise ValueError(f"Unknown SHAP variant: {args.shap_variant}")
    x = jnp.ones(total_patches, dtype=sample.dtype)

    match args.bound_method:
        case "bab":
            bounds_method = shapley_bab
        case _:
            raise ValueError(f"Unknown bound method: {args.bound_method}")

    bounds = []
    rounds = range(args.max_iters) if args.max_iters is not None else it.count()
    for i, (lb, ub) in zip(
        rounds, bounds_method(value_fn, x, in_feature), strict=False
    ):
        print(f"{i}:", lb, ub)
        bounds.append((lb, ub))

    if args.out_file is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        out_file = (
            local_output_dir
            / f"mnist_{args.model}_{in_feature}_{out_feature}_shap_{args.shap_variant}_{args.bound_method}_{timestamp}.csv"
        )
    else:
        out_file = Path(args.out_file)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    bounds = pd.DataFrame(bounds, columns=["lb", "ub"])
    bounds.to_csv(out_file, index=False)
