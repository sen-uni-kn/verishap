# Copyright 2025 David Boetius
import argparse
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import shap
from tqdm import tqdm

from .. import shaplib
from .models import MLP

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
        "--estimator",
        type=str,
        default="KernelSHAP",
        help="The SHAP estimator to use.",
    )
    parser.add_argument(
        "--num-samples",
        type=str,
        default="1000",
        help="The number of validation samples to use for the SHAP estimator. "
        "This can be a single number, a range, or a list of numbers. "
        "For example, --num-samples 1:100:10 specifies a range and "
        "--num-samples 100,200,300 specifies a list.",
    )
    parser.add_argument(
        "--shap-variant",
        type=str,
        default="zero-baseline",
        help="The SHAP variant to use.",
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
    data, targets = getattr(shap.datasets, dataset)()
    data = data.to_numpy().astype(np.float32)
    data = np.nan_to_num(data)

    model = MLP.load(args.model)
    model = eqx.nn.inference_mode(model)
    model = jax.vmap(model, axis_name="batch")

    in_feature = args.feature
    out_feature = args.output_feature
    x = data[args.input]

    num_samples = args.num_samples
    if ":" in num_samples:
        num_samples = range(*map(int, num_samples.split(":")))
    elif "," in num_samples:
        num_samples = list(map(int, num_samples.split(",")))
    else:
        num_samples = [int(num_samples)]

    match args.estimator.lower().replace("-", "").replace("_", "").replace(" ", ""):
        case "exactshap":
            estimator = partial(shaplib.exact_shap, model, silent=True)
        case "kernelshap":
            estimator = partial(shaplib.kernel_shap, model, silent=True)
        case "permutationshap":
            estimator = partial(shaplib.permutation_shap, model, silent=True)
        case "samplingshap":
            estimator = partial(shaplib.sampling_shap, model, silent=True)
        case "leverageshap":
            # TODO :)
            pass
        case _:
            raise ValueError(f"Unknown SHAP estimator: {args.estimator}")

    match args.shap_variant:
        case "zero-baseline":
            baseline = jnp.zeros_like(x)
            estimator = partial(estimator, baseline)
        case _:
            raise ValueError(f"Unknown SHAP variant: {args.shap_variant}")

    estimates = []
    for n in tqdm(num_samples):
        estimate = estimator(x, num_samples=n)
        estimate = estimate[in_feature, out_feature]
        estimates.append({"num_samples": n, "estimate": estimate.item()})

    print("Best Estimate: ", estimates[-1]["estimate"])

    if args.out_file is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        out_file = (
            local_output_dir
            / f"{dataset}"
            / f"{args.model.stem}_{in_feature}_{out_feature}_shap_{args.estimator}_{args.shap_variant}_{timestamp}.csv"
        )
    else:
        out_file = Path(args.out_file)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    estimates = pd.DataFrame(estimates)
    estimates.to_csv(out_file, index=False)
