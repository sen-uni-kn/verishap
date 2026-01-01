# Copyright 2025 David Boetius
import itertools as it
from pathlib import Path

from shap_bounds.logger import ConsoleLogger, FileLogger, JoinLoggers

from .. import runstats
from ..boundstats import BoundStats
from .argument_parser import VisionPatchesCmdArgs
from .overlay_logger import VisionPatchesBoundsLogger

local_output_dir = Path(__file__).parent / "output"


if __name__ == "__main__":
    args = (
        VisionPatchesCmdArgs()
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
    time_stats = {}
    iter_stats = {}

    out_dir = args.out_dir(local_output_dir)
    overlay_logger = None
    if args.overlay_logger:
        overlay_logger = VisionPatchesBoundsLogger(
            args.sample,
            args.num_patches[0],
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

        with BoundStats() as bound_stats:
            bounds_iter = args.bound_method(logger)()
            iters = range(args.max_iters) if args.max_iters is not None else it.count()
            for i, (lb, ub) in zip(iters, bounds_iter, strict=False):
                # The logger shows the bounds, don't have to print here.
                runtime = bound_stats.record(i, lb, ub)

                if args.timeout is not None and runtime > args.timeout:
                    print(f"Timeout reached after {runtime:.2f} seconds.")
                    time_stats["timeout"] = True
                    iter_stats["timeout"] = i
                    break

        if "timeout" not in time_stats and args.timeout is not None:
            time_stats["timeout"] = False

        time_stats["overall"] = bound_stats.runtime
        logger.log_stats(
            "overall",
            {
                "runtime": time_stats | bound_stats.time_stats,
                "iterations": iter_stats | bound_stats.iter_stats,
            },
        )
