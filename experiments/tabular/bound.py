# Copyright 2025 David Boetius
import itertools as it
from pathlib import Path
from time import perf_counter

import jax.numpy as jnp

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
        .bound_method_args()
        .timeout_args()
        .logger_args()
        .out_args()
        .parse_args()
    )
    in_feature = args.feature

    loggers = [] if args.silent else [ConsoleLogger()]
    timer = Timer()
    time_stats = {}
    iter_stats = {}

    with FileLogger(args.out_dir(local_output_dir)) as file_logger:
        logger = JoinLoggers(*loggers, file_logger)
        logger.log_config("run_details", machine_and_code_details())
        logger.log_config("cmd_args", args.all_arguments)

        with timer["overall"] as timer_context:
            bounds_iter = args.bound_method(logger)()
            iters = range(args.max_iters) if args.max_iters is not None else it.count()
            for i, (lb, ub) in zip(iters, bounds_iter, strict=False):
                # The logger shows the bounds, don't have to print here.
                # Now log the time it takes for features to be separated.
                runtime = perf_counter() - timer_context.start

                lb, ub = lb.flatten(), ub.flatten()
                lb_vs_each_ub = jnp.reshape(lb, (-1, 1)) >= jnp.reshape(ub, (1, -1))
                ub_vs_each_lb = jnp.reshape(ub, (-1, 1)) <= jnp.reshape(lb, (1, -1))
                separated = lb_vs_each_ub.any(axis=-1)
                largest = lb_vs_each_ub.all(axis=-1)
                smallest = ub_vs_each_lb.all(axis=-1)
                if separated.any() and "some_separated" not in time_stats:
                    runtime = perf_counter() - timer_context.start
                    print(f"Some features separated after {runtime:.2f} seconds.")
                    time_stats["some_separated"] = runtime
                    iter_stats["some_separated"] = i
                if separated.all() and "all_separated" not in time_stats:
                    print(f"All features separated after {runtime:.2f} seconds.")
                    time_stats["all_separated"] = runtime
                    iter_stats["all_separated"] = i
                if largest.any() and "largest" not in time_stats:
                    print(f"Single most influential feature found after {runtime:.2f} seconds.")
                    time_stats["largest_shap"] = runtime
                    iter_stats["largest_shap"] = i
                if smallest.any() and "smallest" not in time_stats:
                    print(f"Single least influential feature found after {runtime:.2f} seconds.")
                    time_stats["smallest_shap"] = runtime
                    iter_stats["smallest_shap"] = i

                if args.timeout is not None and runtime > args.timeout:
                    print(f"Timeout reached after {runtime:.2f} seconds.")
                    time_stats["timeout"] = True
                    iter_stats["timeout"] = i
                    break

        if "timeout" not in time_stats and args.timeout is not None:
            time_stats["timeout"] = False

        time_stats["overall"] = timer_context.runtime
        logger.log_stats("overall", {"runtime": time_stats, "iterations": iter_stats})
