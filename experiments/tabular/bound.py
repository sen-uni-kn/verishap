# Copyright 2025 David Boetius
import itertools as it
from pathlib import Path

import numpy as np
import pandas as pd

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
        .bound_method_args()
        .out_file_args()
        .parse_args()
    )
    np.random.seed(0)
    model = args.model
    in_feature = args.feature
    data, _ = args.dataset
    x = data[args.input]
    value_fn = args.value_function(model, x, data)
    bounds_method = args.bounds_method

    bounds_iter = bounds_method(value_fn, x, in_feature)
    iters = range(args.max_iters) if args.max_iters is not None else it.count()
    bounds = [(lb, ub) for (lb, ub), _ in zip(bounds_iter, iters, strict=False)]
    print("Best Bound: ", bounds[-1][0], bounds[-1][1])

    bounds = pd.DataFrame(bounds, columns=["lb", "ub"])
    bounds.to_csv(args.out_file(local_output_dir), index=False)
