# Copyright 2025 David Boetius
from pathlib import Path

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
    estimator = args.estimator()
    in_feature, out_feature = args.feature, args.output_feature
    num_samples = args.num_samples

    estimates = []
    for n in tqdm(num_samples):
        estimate = estimator(num_samples=n)
        estimate = estimate[in_feature, out_feature]
        estimates.append({"num_samples": n, "estimate": estimate.item()})

    estimates = pd.DataFrame(estimates)
    print(estimates)
    print("Best Estimate: ", estimates.iloc[-1]["estimate"])

    estimates.to_csv(args.out_file(local_output_dir), index=False)
