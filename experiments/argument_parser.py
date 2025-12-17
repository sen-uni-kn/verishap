# Copyright 2025 David Boetius
"""Utilities for parsing command line arguments."""

import argparse
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torchvision

from shap_bounds import (
    baseline_value,
    marginal_value,
    multi_shap_bab,
    superfeature_baseline_value,
    superfeature_marginal_value,
)
from shap_bounds.logger import Logger

from . import shaplib
from .datasets import load_dataset
from .leverageshap import leverage_shap
from .models import CNN, MLP


class NumpyVisionDataset:
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        shape: tuple[int, ...],
        mid = 0.0,
        ran = 1.0,
        channel_last = False,
    ):
        self.dataset = dataset
        self.shape = shape
        self.mid = mid
        self.ran = ran
        self.channel_last = channel_last

    def __getitem__(self, index) -> np.ndarray:
        data = self.dataset.data[index]
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        assert isinstance(data, np.ndarray)
        if self.channel_last:
            if data.ndim == 3:
                data = np.moveaxis(data, -1, 0)
            else:
                data = np.moveaxis(data, -1, 0)
        if self.shape[0] == 1 and data.ndim == 2:
            batch_shape = data.shape[:-2]
        else:
            batch_shape = data.shape[:-3]
        data = data.reshape(*batch_shape, *self.shape)
        data = data / 255.0
        data = (data - self.mid) / self.ran
        return data

    def __iter__(self) -> Iterator[np.ndarray]:
        for index in range(len(self.dataset)):
            yield self[index]

    def __len__(self) -> int:
        return len(self.dataset)


class CmdArgs:
    def __init__(self, *parser_args, **parser_kwargs):
        np.random.seed(0)
        torch.manual_seed(0)
        self.parser = argparse.ArgumentParser(*parser_args, **parser_kwargs)
        self.args = None

    def model_args(self, default_model: Path | None = None) -> "CmdArgs":
        default = (
            {"required": True} if default_model is None else {"default": default_model}
        )
        self.parser.add_argument(
            "--model",
            type=Path,
            help="The path of the model file to load.",
            **default,
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
            default=None,
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
            help="The SHAP variant to use. Options: zero-baseline, marginal, "
            "superfeature-zero-baseline, superfeature-marginal",
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
        self.parser.add_argument(
            "--seed",
            default=0,
            type=int,
            help="The random seed to use for the experiment.",
        )
        return self

    def timeout_args(self) -> "CmdArgs":
        self.parser.add_argument(
            "--timeout",
            type=float,
            default=None,
            help="The timeout in seconds for the experiment.",
        )
        return self

    def logger_args(self) -> "CmdArgs":
        self.parser.add_argument(
            "--silent",
            action="store_true",
            help="Do not print any output to the console.",
        )
        return self

    def out_args(self) -> "CmdArgs":
        self.parser.add_argument(
            "--out",
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
    def all_arguments(self) -> dict:
        return {key: str(value) for key, value in vars(self.args).items()}

    @property
    def data(self) -> Iterable[np.ndarray]:
        dataset = self.args.dataset
        if dataset is None:
            dataset = self.args.model.stem.split("-")[0]
        if dataset.lower() == "mnist":
            testset = torchvision.datasets.MNIST(
                ".datasets",
                train=False,
                download=True,
                transform=torchvision.transforms.ToTensor(),
            )
            return NumpyVisionDataset(testset, shape=(1, 28, 28))
        elif dataset.lower() == "cifar10":
            testset = torchvision.datasets.CIFAR10(
                ".datasets",
                train=False,
                download=True,
                transform=torchvision.transforms.ToTensor(),
            )
            return NumpyVisionDataset(testset, shape=(3, 32, 32))
        else:
            data, _ = load_dataset(dataset)
        return data

    @property
    def model(self) -> Callable:
        if "cnn" in self.args.model.stem.lower():
            model_, state = CNN.load(self.args.model)
            model_ = eqx.nn.inference_mode(model_)

            @partial(jax.vmap, axis_name="batch")
            def model(x):
                y, _ = model_(x, state)
                return y

            return model
        else:
            model = MLP.load(self.args.model)
            model = eqx.nn.inference_mode(model)
            model = jax.vmap(model, axis_name="batch")
            return model

    @property
    def masks(self) -> np.ndarray | None:
        """Obtain masks for superfeature value functions.

        Returns None if superfeatures are not supported.
        """
        return None

    @property
    def sample(self) -> np.ndarray:
        return self.data[self.input]

    @property
    def background_data(self) -> np.ndarray:
        num_background = self.num_background_samples
        data = self.data
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(data))
        return data[perm[:num_background]]

    @property
    def data_mean(self) -> np.ndarray:
        return self.data[:].mean(axis=0)

    @property
    def base_mask(self) -> np.ndarray:
        return jnp.ones_like(self.sample)

    @property
    def value_function(self) -> Callable:
        shap_variant = self.shap_variant
        out_feature = self.output_feature

        masks = self.masks
        if masks is None and "superfeature" in shap_variant:
            raise NotImplementedError(
                "Superfeature value functions are not implemented for this experiment."
            )

        x = self.sample
        match shap_variant:
            case "zero-baseline":
                baseline = jnp.zeros_like(x)
                return baseline_value(self.model, x, baseline, out_feature)
            case "mean-baseline":
                baseline = self.data_mean.reshape(x.shape)
                return baseline_value(self.model, x, baseline, out_feature)
            case "marginal":
                return marginal_value(self.model, x, self.background_data, out_feature)
            case "superfeature-zero-baseline":
                baseline = jnp.zeros_like(x)
                return superfeature_baseline_value(
                    self.model, x, baseline, masks, out_feature
                )
            case "superfeature-mean-baseline":
                baseline = self.data_mean.reshape(x.shape)
                return superfeature_baseline_value(
                    self.model, x, baseline, masks, out_feature
                )
            case "superfeature-marginal":
                return superfeature_marginal_value(
                    self.model, x, self.background_data, masks, out_feature
                )
            case _:
                raise ValueError(f"Unknown SHAP variant: {self.shap_variant}")

    def bound_method(self, logger: Logger) -> Callable:
        match self.args.bound_method.lower():
            case "bab":
                res = partial(
                    multi_shap_bab,
                    self.value_function,
                    self.base_mask,
                    self.feature,
                    log=logger,
                    **self.bound_kwargs,
                )
                return res
            case _:
                raise ValueError(f"Unknown bound method: {self.args.bound_method}")

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
        shap_variant = self.shap_variant
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
                estimator = partial(
                    leverage_shap,
                    self.model,
                )
            case _:
                raise ValueError(f"Unknown SHAP estimator: {self.args.estimator}")

        masks = self.masks
        if masks is None and "superfeature" in shap_variant:
            raise NotImplementedError(
                "Superfeature value functions are not implemented for this experiment."
            )

        match shap_variant:
            case "zero-baseline":
                baseline = jnp.zeros_like(self.sample)
                estimator = partial(estimator, baseline, self.sample)
            case "mean-baseline":
                baseline = self.data_mean.reshape(self.sample.shape)
                estimator = partial(estimator, baseline, self.sample)
            case "marginal":
                estimator = partial(estimator, self.background_data, self.sample)
            case "superfeature-zero-baseline" | "superfeature-mean-baseline":
                if shap_variant == "superfeature-zero-baseline":
                    baseline = jnp.zeros_like(self.sample)
                else:
                    baseline = self.data_mean.reshape(self.sample.shape)
                baseline = jnp.expand_dims(baseline, axis=0)
                masker = shaplib.superfeature_masker(self.sample, baseline, masks)
                x = jnp.ones(masks.shape[0], dtype=jnp.float32)
                estimator = partial(estimator, masker, x)
            case "superfeature-marginal":
                masker = shaplib.superfeature_masker(
                    self.sample, self.background_data, masks
                )
                estimator = partial(estimator, masker, self.sample)
            case _:
                raise ValueError(f"Unknown SHAP variant: {self.args.shap_variant}")

        seed = self.args.seed
        np.random.seed(seed)
        torch.manual_seed(seed + 1)
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

    def _unique_name(self) -> str:
        name = []
        if hasattr(self.args, "model"):
            name.append(self.args.model.stem)
        if hasattr(self.args, "feature"):
            if self.args.feature is None:
                name.append("all-features")
            else:
                name.append(f"{self.args.feature}")
        if hasattr(self.args, "output_feature"):
            name.append(f"{self.args.output_feature}")
        name.append(self.method_name)
        if hasattr(self.args, "shap_variant"):
            name.append(self.args.shap_variant)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        name.append(timestamp)
        return "_".join(name)

    def out_file(self, local_output_dir: Path) -> Path:
        if self.args.out is None:
            out_file = local_output_dir / (self._unique_name() + ".csv")
        else:
            out_file = Path(self.args.out)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        return out_file

    def out_dir(self, local_output_dir: Path) -> Path:
        if self.args.out is None:
            out_file = local_output_dir / self._unique_name()
        else:
            out_file = Path(self.args.out)
        out_file.mkdir(parents=True, exist_ok=True)
        return out_file

    def __getattr__(self, name: str) -> Any:
        if self.args is None:
            raise AttributeError("Arguments not parsed yet. Call `parse_args()` first.")
        return getattr(self.args, name)
