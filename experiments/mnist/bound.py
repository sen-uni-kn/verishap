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
    in_feature = args.feature

    bounds_iter = args.bound_method()
    iters = range(args.max_iters) if args.max_iters is not None else it.count()
    bounds = []
    for (lb, ub), _ in zip(bounds_iter, iters, strict=False):
        if in_feature is not None:
            bounds.append({"lb": lb.item(), "ub": ub.item()})
        else:
            bounds.append(
                {(i, "lb"): lb_.item() for i, lb_ in enumerate(lb)}
                | {(i, "ub"): ub_.item() for i, ub_ in enumerate(ub)}
            )

    if in_feature is not None:
        bounds = pd.DataFrame(bounds)
    else:
        columns = pd.MultiIndex.from_product(
            [[i for i in range(len(lb))], ["lb", "ub"]], names=["feature", "bound"]
        )
        bounds = pd.DataFrame(bounds, columns=columns)
    print(bounds)
    print("Best Bounds:")
    if in_feature is not None:
        print(bounds.iloc[-1])
    else:
        print(bounds.iloc[-1].unstack(level=1))

    bounds.to_csv(args.out_file(local_output_dir), index="iteration")
