# Copyright 2025 David Boetius
from functools import partial
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from shap_bounds.logger import ConsoleLogger, FileLogger, JoinLoggers
from shap_bounds.timer import Timer

from ..runstats import machine_and_code_details
from .argument_parser import TabularCmdArgs
from .. import shaplib

local_resoure_dir = Path(__file__).parent / "resources"
local_output_dir = Path(__file__).parent / "output"


if __name__ == "__main__":
    args = (
        TabularCmdArgs()
        .model_args(local_resoure_dir)
        .dataset_args()
        .feature_args()
        .shap_variant_args()
        .out_args()
        .parse_args()
    )
    estimator = partial(
        shaplib.exact_shap,
        args.model,
        silent=args.silent,
    )
    in_feature, out_feature = args.feature, args.output_feature
    num_samples = 9223372036854775807

    loggers = [] if args.silent else [ConsoleLogger()]
    timer = Timer()

    with FileLogger(args.out_dir(local_output_dir)) as file_logger:
        logger = JoinLoggers(*loggers, file_logger)
        logger.log_config("run_details", machine_and_code_details())
        logger.log_config("cmd_args", args.args.vars())

        with timer["estimate"] as timer_context:
            estims = estimator(num_samples=num_samples)

        if in_feature is None:
            estims = estims[:, out_feature].squeeze()
            estims = {i: v for i, v in enumerate(estims)}
        else:
            estim = estims[in_feature, out_feature]
            estims = {in_feature: estim}

        logger.log_stats("overall", {"runtime": timer_context.runtime, "estimates": estims})
