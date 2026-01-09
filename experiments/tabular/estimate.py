# Copyright 2025 David Boetius
from collections import defaultdict
from pathlib import Path

import jax
import numpy as np
import torch
from tqdm import tqdm

from shap_bounds.logger import ConsoleLogger, FileLogger, JoinLoggers
from shap_bounds.timer import Timer

from ..runstats import machine_and_code_details
from .argument_parser import TabularCmdArgs

local_output_dir = Path(__file__).parent / "output"


class TimeoutError(Exception):
    pass


if __name__ == "__main__":
    args = (
        TabularCmdArgs()
        .model_args()
        .dataset_args()
        .feature_args()
        .shap_variant_args()
        .estimator_args()
        .timeout_args()
        .logger_args()
        .out_args()
        .parse_args()
    )
    estimator = args.estimator()
    in_feature, out_feature = args.feature, args.output_feature
    num_samples = args.num_samples

    loggers = [] if args.silent else [ConsoleLogger()]
    timer = Timer()

    seeds = [args.seed] if args.random_seeds is None else args.random_seeds
    with FileLogger(args.out_dir(local_output_dir)) as file_logger:
        logger = JoinLoggers(*loggers, file_logger)
        logger.log_config("run_details", machine_and_code_details())
        logger.log_config("cmd_args", args.all_arguments)
        logger.log_config("further_stats", args.further_run_stats)

        runtimes = defaultdict(list)

        try:
            progress = tqdm(total=len(seeds) * len(num_samples))
            for n in num_samples:
                for seed in seeds:
                    np.random.seed(seed)
                    torch.manual_seed(seed + 1)
                    with timer["estimate"] as timer_context:
                        estims = estimator(num_samples=n)
                    runtime = timer_context.runtime

                    if args.timeout is not None and runtime > args.timeout:
                        raise TimeoutError()

                    if in_feature is None:
                        estims = estims[:, out_feature].squeeze()
                        estims = {f"{i}": v.item() for i, v in enumerate(estims)}
                    else:
                        estim = estims[in_feature, out_feature]
                        estims = {f"{in_feature}": estim.item()}

                    logger.log_iter_stats(
                        "estimate", (n, seed), {"runtime": runtime}, **estims
                    )
                    runtimes[n].append(runtime)
                    progress.update(1)
        except jax.errors.RuntimeError:  # catch out of memory errors
            print("Out of memory error. Stopping experiment.")
        except TimeoutError:
            print(f"Timeout reached for {n} samples. Stopping experiment.")
        finally:
            logger.log_stats(
                "overall", {"runtimes": runtimes}
            )
