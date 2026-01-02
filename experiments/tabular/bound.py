# Copyright 2025 David Boetius
import itertools as it
from pathlib import Path
from time import perf_counter

from shap_bounds.logger import ConsoleLogger, FileLogger, JoinLoggers

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
        .bound_method_args()
        .timeout_args()
        .logger_args()
        .out_args()
        .parse_args()
    )
    in_feature = args.feature

    loggers = [] if args.silent else [ConsoleLogger()]
    stats = {"timeout": False, "max_iters": False}

    with FileLogger(args.out_dir(local_output_dir)) as file_logger:
        logger = JoinLoggers(*loggers, file_logger)
        logger.log_config(
            "run_details", machine_and_code_details() | {"sample": args.sample.tolist()}
        )
        logger.log_config("cmd_args", args.all_arguments)

        bounds_iter = args.bound_method(logger)()
        iters = it.count()
        start_time = perf_counter()
        for i, _ in zip(iters, bounds_iter, strict=False):
            # The logger shows the bounds, don't have to print here.
            runtime = perf_counter() - start_time
            if args.timeout is not None and runtime > args.timeout:
                print(f"Timeout reached after {runtime:.2f} seconds.")
                stats["timeout"] = True
                break
            if i == args.max_iters:
                print("Maximum number of iterations reached.")
                stats["max_iters"] = True
                break

        logger.log_stats("overall", {"runtime": runtime, "iterations": i, **stats})
