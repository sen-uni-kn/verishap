# Copyright 2025 David Boetius
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .argument_parser import TabularCmdArgs

local_resoure_dir = Path(__file__).parent / "resources"
local_output_dir = Path(__file__).parent / "output"


if __name__ == "__main__":
    args = (
        TabularCmdArgs()
        .model_args(local_resoure_dir)
        .dataset_args()
        .feature_args()
        .shap_variant_args()
        .estimator_args()
        .out_file_args()
        .parse_args()
    )
    np.random.seed(0)
    model = args.model
    in_feature, out_feature = args.feature, args.output_feature
    data, _ = args.dataset
    x = data[args.input]
    value_fn = args.value_function(model, x, data)
    estimator = args.estimator(model, x, data)
    num_samples = args.num_samples

    estimates = []
    for n in tqdm(num_samples):
        estimate = estimator(x, num_samples=n)
        estimate = estimate[in_feature, out_feature]
        estimates.append({"num_samples": n, "estimate": estimate.item()})

    print("Best Estimate: ", estimates[-1]["estimate"])

    estimates = pd.DataFrame(estimates)
    estimates.to_csv(args.out_file(local_output_dir), index=False)
