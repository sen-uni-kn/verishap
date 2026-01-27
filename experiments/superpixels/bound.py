# Copyright 2025 David Boetius
import itertools as it
from pathlib import Path
from time import perf_counter

from shap_bounds.logger import ConsoleLogger, FileLogger, JoinLoggers

from .. import runstats
from .argument_parser import SuperpixelsCmdArgs
from .overlay_logger import SuperpixelsBoundsLogger

local_resoure_dir = Path(__file__).parent / "resources"
local_output_dir = Path(__file__).parent / "output"


if __name__ == "__main__":
    args = (
        SuperpixelsCmdArgs()
        .model_args()
        .dataset_args()
        .segmentation_args()
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

    out_dir = args.out_dir(local_output_dir)
    overlay_logger = None
    if args.overlay_logger:
        overlay_logger = SuperpixelsBoundsLogger(
            args.sample,
            args.masks,
            out_dir,
            feature=in_feature,
            show=not args.silent,
        )

    with FileLogger(out_dir) as file_logger:
        logger_parts = [*loggers, file_logger]
        if overlay_logger is not None:
            logger_parts.append(overlay_logger)
        logger = JoinLoggers(*logger_parts)
        logger.log_config("run_details", runstats.machine_and_code_details())
        logger.log_config("cmd_args", args.all_arguments)
        logger.log_config("further_stats", args.further_run_stats)

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
