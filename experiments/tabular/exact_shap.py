# Copyright 2025 David Boetius
from functools import partial
from pathlib import Path

from shap_bounds.logger import ConsoleLogger, FileLogger, JoinLoggers
from shap_bounds.timer import Timer

from .. import shaplib
from ..runstats import machine_and_code_details
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
        .logger_args()
        .out_args()
        .parse_args()
    )
    args.args.estimator = "ExactSHAP"
    args.args.seed = 0
    estimator = args.estimator()
    in_feature, out_feature = args.feature, args.output_feature
    num_samples = 9223372036854775807

    loggers = [] if args.silent else [ConsoleLogger()]
    timer = Timer()

    with FileLogger(args.out_dir(local_output_dir)) as file_logger:
        logger = JoinLoggers(*loggers, file_logger)
        logger.log_config("run_details", machine_and_code_details())
        logger.log_config("cmd_args", args.all_arguments)

        with timer["estimate"] as timer_context:
            estims = estimator(num_samples=num_samples)

        if in_feature is None:
            estims = estims[:, out_feature].squeeze()
            estims = {i: v.item() for i, v in enumerate(estims)}
        else:
            estim = estims[in_feature, out_feature]
            estims = {in_feature: estim.item()}

        logger.log_stats(
            "overall", {"runtime": timer_context.runtime, "estimates": estims}
        )
