# Copyright 2025 David Boetius
"""Utilities for parsing command line arguments."""

import argparse
from abc import ABC, abstractmethod
from collections.abc import Iterable
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np
import pandas as pd
import torch

from shap_bounds import baseline_value, marginal_value, shapley_bab

from . import shaplib


class CmdArgs(ABC):
    def __init__(self, *parser_args, **parser_kwargs):
        np.random.seed(0)
        torch.manual_seed(0)
        self.parser = argparse.ArgumentParser(*parser_args, **parser_kwargs)
        self.args = None

    def model_args(self, resource_dir: Path) -> "CmdArgs":
        self.parser.add_argument(
            "--model",
            type=Path,
            required=True,
            help="The path of the model file to load.",
        )
        return self

    def dataset_args(
        self, default_dataset: str = None, default_data_index: int = 0
    ) -> "CmdArgs":
        self.parser.add_argument(
            "--dataset",
            type=str,
            default=default_dataset,
            help="The name of the dataset to use.",
        )
        self.data_index_args(default_data_index)
        return self

    def data_index_args(self, default: int = 0) -> "CmdArgs":
        self.parser.add_argument(
            "--input",
            type=int,
            default=default,
            help="The index of the input sample to analyse in the dataset.",
        )
        return self

    def feature_args(self) -> "CmdArgs":
        self.parser.add_argument(
            "--feature",
            type=int,
            default=0,
            help="The index of the input feature to compute SHAP bounds for.",
        )
        self.parser.add_argument(
            "--output-feature",
            type=int,
            default=0,
            help="The index of the output feature to explain.",
        )
        return self

    def shap_variant_args(self) -> "CmdArgs":
        self.parser.add_argument(
            "--shap-variant",
            type=str,
            default="zero-baseline",
            help="The SHAP variant to use. Options: zero-baseline, marginal",
        )
        self.parser.add_argument(
            "--num-background-samples",
            type=int,
            default=100,
            help="The number of background samples to use for the marginal SHAP variant.",
        )
        return self

    def bound_method_args(self) -> "CmdArgs":
        self.parser.add_argument(
            "--bound-method",
            type=str,
            default="bab",
            help="The method to use for computing the bounds.",
        )
        self.parser.add_argument(
            "--bound-options",
            type=str,
            default="",
            help="Keyword arguments to pass to the bound method in the format "
            "key1=value1,key2=value2,...",
        )
        self.parser.add_argument(
            "--max-iters",
            type=int,
            default=None,
            help="How many iterations to perform at most.",
        )
        return self

    def estimator_args(self) -> "CmdArgs":
        self.parser.add_argument(
            "--estimator",
            type=str,
            default="KernelSHAP",
            help="The SHAP estimator to use. "
            "Options: ExactSHAP, KernelSHAP, PermutationSHAP, SamplingSHAP, LeverageSHAP.",
        )
        self.parser.add_argument(
            "--num-samples",
            type=str,
            default="1000",
            help="The number of validation samples to use for the SHAP estimator. "
            "This can be a single number, a range, or a list of numbers. "
            "For example, --num-samples 1:100:10 specifies a range and "
            "--num-samples 100,200,300 specifies a list.",
        )
        return self

    def out_file_args(self) -> "CmdArgs":
        self.parser.add_argument(
            "--out-file",
            type=str,
            default=None,
            help="Where to save the experiment results.",
        )
        return self

    def parse_args(self) -> "CmdArgs":
        self.args = self.parser.parse_args()
        return self

    # =========================================================================

    @property
    @abstractmethod
    def model(self) -> Callable:
        """Obtain the model as a callable."""
        raise NotImplementedError

    @property
    @abstractmethod
    def data(self) -> Iterable[np.ndarray]:
        """Obtain the data as a numpy array or indexable dataset."""
        raise NotImplementedError

    @property
    def sample(self) -> np.ndarray:
        return self.data[self.args.input]

    @property
    def background_data(self) -> np.ndarray:
        num_background = self.args.num_background_samples
        data = self.data
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(data))
        return data[perm[:num_background]]

    @property
    def value_function(self) -> Callable:
        out_feature = self.args.output_feature
        match self.shap_variant:
            case "zero-baseline":
                baseline = jnp.zeros_like(self.sample)
                return baseline_value(self.model, baseline, out_feature)
            case "marginal":
                return marginal_value(self.model, self.background_data, out_feature)
            case _:
                raise ValueError(f"Unknown SHAP variant: {self.shap_variant}")

    @property
    def bounds_method(self) -> Callable:
        match self.bound_method:
            case "bab":
                return partial(
                    shapley_bab,
                    self.value_function,
                    self.sample,
                    self.feature,
                    **self.bound_kwargs,
                )
            case _:
                raise ValueError(f"Unknown bound method: {self.bound_method}")

    @property
    def bound_kwargs(self) -> dict:
        bound_kwargs = {}
        if self.bound_options:
            for option in self.bound_options.split(","):
                k, v = option.split("=")
                try:
                    v = eval(v, {})
                except NameError:
                    pass
                bound_kwargs[k] = v
        return bound_kwargs

    def estimator(self) -> Callable:
        match (
            self.args.estimator.lower()
            .replace("-", "")
            .replace("_", "")
            .replace(" ", "")
        ):
            case "exactshap":
                estimator = partial(
                    shaplib.exact_shap,
                    self.model,
                    silent=True,
                )
            case "kernelshap":
                estimator = partial(
                    shaplib.kernel_shap,
                    self.model,
                    silent=True,
                )
            case "permutationshap":
                estimator = partial(
                    shaplib.permutation_shap,
                    self.model,
                    silent=True,
                )
            case "samplingshap":
                estimator = partial(
                    shaplib.sampling_shap,
                    self.model,
                    silent=True,
                )
            case "leverageshap":
                # TODO :)
                pass
            case _:
                raise ValueError(f"Unknown SHAP estimator: {self.args.estimator}")

        match self.args.shap_variant:
            case "zero-baseline":
                baseline = jnp.zeros_like(self.sample)
                estimator = partial(estimator, baseline, self.sample)
            case "marginal":
                estimator = partial(estimator, self.background_data, self.sample)
            case _:
                raise ValueError(f"Unknown SHAP variant: {self.args.shap_variant}")

        return estimator

    @property
    def num_samples(self) -> list:
        num_samples = self.args.num_samples
        if ":" in num_samples:
            num_samples = list(range(*map(int, num_samples.split(":"))))
        elif "," in num_samples:
            num_samples = list(map(int, num_samples.split(",")))
        else:
            num_samples = [int(num_samples)]
        return num_samples

    @property
    def method_name(self) -> str:
        if hasattr(self.args, "bound_method"):
            return self.args.bound_method
        elif hasattr(self.args, "estimator"):
            return self.args.estimator
        else:
            raise ValueError("No method name found.")

    def out_file(self, local_output_dir: Path) -> Path:
        if self.args.out_file is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
            out_file = local_output_dir / (
                f"{self.args.model.stem}_{self.args.feature}_{self.args.output_feature}_shap"
                f"_{self.method_name}_{self.args.shap_variant}_{timestamp}.csv"
            )
        else:
            out_file = Path(self.args.out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        return out_file

    def __getattr__(self, name: str) -> Any:
        if self.args is None:
            raise AttributeError("Arguments not parsed yet. Call `parse_args()` first.")
        return getattr(self.args, name)
