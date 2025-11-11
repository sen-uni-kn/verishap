# Copyright 2025 David Boetius
import argparse
import itertools as it
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from shap_bounds import baseline_value, shapley_bab

from .models import MLP, SumOut

local_output_dir = Path(__file__).parent / "output"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--mlp",
        action="store_true",
        help="Use an MLP model.",
    )
    model_group.add_argument(
        "--sum",
        action="store_true",
        help="Use a trivial linear model.",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=10,
        help="The dimension of the input.",
    )
    parser.add_argument(
        "--feature",
        type=int,
        default=0,
        help="The index of the input feature to compute SHAP bounds for.",
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

    np.random.seed(0)

    in_dim = args.input_dim
    if args.mlp:
        model = MLP(in_dim, [100, 100], key=jax.random.PRNGKey(0))
    elif args.sum:
        model = SumOut()
    else:
        raise ValueError(f"Unknown model: {args.model}")
    model = jax.vmap(model)

    # x = np.random.uniform(0.0, 1 / in_dim, (in_dim,))
    x = np.random.randn(in_dim)
    x = x * (1 / in_dim)
    in_feature = args.feature

    match args.shap_variant:
        case "zero-baseline":
            baseline = jnp.zeros_like(x)
            value_fn = baseline_value(model, baseline, 0)
        case _:
            raise ValueError(f"Unknown SHAP variant: {args.shap_variant}")

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
        bounds.append((lb.item(), ub.item()))

    if args.out_file is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        model_name = "mlp" if args.mlp else "sum"
        out_file = (
            local_output_dir
            / f"{model_name}_{in_dim}_shap_{args.shap_variant}_{args.bound_method}_{timestamp}.csv"
        )
    else:
        out_file = Path(args.out_file)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    bounds = pd.DataFrame(bounds, columns=["lb", "ub"])
    bounds.to_csv(out_file, index=False)
