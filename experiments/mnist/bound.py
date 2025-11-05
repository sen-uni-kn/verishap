# Copyright 2025 David Boetius
import itertools as it
from pathlib import Path

import pandas as pd

from .argument_parser import MNISTCmdArgs

local_resoure_dir = Path(__file__).parent / "resources"
local_output_dir = Path(__file__).parent / "output"


if __name__ == "__main__":
    args = (
        MNISTCmdArgs()
        .model_args(local_resoure_dir)
        .dataset_args()
        .segmentation_args()
        .feature_args()
        .shap_variant_args()
        .bound_method_args()
        .out_file_args()
        .parse_args()
    )

    bounds_iter = args.bound_method()
    iters = range(args.max_iters) if args.max_iters is not None else it.count()
    bounds = [(lb, ub) for (lb, ub), _ in zip(bounds_iter, iters, strict=False)]
    print("Best Bound: ", bounds[-1][0], bounds[-1][1])

    bounds = pd.DataFrame(bounds, columns=["lb", "ub"])
    bounds.to_csv(args.out_file(local_output_dir), index=False)
