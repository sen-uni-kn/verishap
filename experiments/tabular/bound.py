# Copyright 2025 David Boetius
import itertools as it
from pathlib import Path

import pandas as pd

from shap_bounds.logger import ConsoleLogger, FileLogger, JoinLoggers
from shap_bounds.timer import Timer

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
        .logger_args()
        .out_args()
        .parse_args()
    )
    in_feature = args.feature

    loggers = [] if args.silent else [ConsoleLogger()]
    timer = Timer()

    with FileLogger(args.out_dir(local_output_dir)) as file_logger:
        logger = JoinLoggers(*loggers, file_logger)

        with timer["overall"]:
            bounds_iter = args.bound_method(logger)()
            iters = range(args.max_iters) if args.max_iters is not None else it.count()
            list(zip(bounds_iter, iters, strict=False))  # the logger shows the bounds

        logger.log_stats("overall", {"runtime": timer.last})
