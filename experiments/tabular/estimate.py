# Copyright 2025 David Boetius
from pathlib import Path

from tqdm import tqdm

from shap_bounds.logger import ConsoleLogger, FileLogger, JoinLoggers
from shap_bounds.timer import Timer

from ..runstats import machine_and_code_details
from .argument_parser import TabularCmdArgs

local_output_dir = Path(__file__).parent / "output"


if __name__ == "__main__":
    args = (
        TabularCmdArgs()
        .model_args()
        .dataset_args()
        .feature_args()
        .shap_variant_args()
        .estimator_args()
        .logger_args()
        .out_args()
        .parse_args()
    )
    estimator = args.estimator()
    in_feature, out_feature = args.feature, args.output_feature
    num_samples = args.num_samples

    loggers = [] if args.silent else [ConsoleLogger()]
    timer = Timer()

    with FileLogger(args.out_dir(local_output_dir)) as file_logger:
        logger = JoinLoggers(*loggers, file_logger)
        logger.log_config("run_details", machine_and_code_details())
        logger.log_config("cmd_args", args.all_arguments)

        for n in tqdm(num_samples, disable=args.silent):
            with timer["estimate"] as timer_context:
                estims = estimator(num_samples=n)
            if in_feature is None:
                estims = estims[:, out_feature].squeeze()
                estims = {f"{i}": v.item() for i, v in enumerate(estims)}
            else:
                estim = estims[in_feature, out_feature]
                estims = {f"{in_feature}": estim.item()}

            logger.log_iter_stats(
                "estimate", n, {"runtime": timer_context.runtime}, **estims
            )
