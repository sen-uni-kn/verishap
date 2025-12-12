# Copyright 2025 David Boetius
"""Utilities for parsing command line arguments for CIFAR10 experiments."""

from collections.abc import Iterable
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Callable, Iterator

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torchvision

from ..argument_parser import CmdArgs
from .cnn import CNN
from .resnet import resnet18


class NumpyDataset:
    def __init__(self, dataset: torch.utils.data.Dataset):
        self.dataset = dataset

    def __getitem__(self, index) -> np.ndarray:
        data = self.dataset.data[index].numpy()
        data = data.reshape(*data.shape[:-2], 1, 28, 28)
        data = data / 255.0
        return data

    def __iter__(self) -> Iterator[np.ndarray]:
        for index in range(len(self.dataset)):
            yield self[index]

    def __len__(self) -> int:
        return len(self.dataset)


class CIFAR10CmdArgs(CmdArgs):
    def __init__(self, *parser_args, **parser_kwargs):
        super().__init__(*parser_args, **parser_kwargs)

    def dataset_args(
        self, default_dataset: str = None, default_data_index: int = 0
    ) -> "CIFAR10CmdArgs":
        return self

    def data_index_args(self, default: int = 0) -> "CIFAR10CmdArgs":
        return self

    def feature_args(self) -> "CmdArgs":
        self.parser.add_argument(
            "--feature",
            type=int,
            default=None,
            help="The index of the input feature to compute SHAP bounds for.",
        )
        return self

    def segmentation_args(self) -> "CIFAR10CmdArgs":
        self.parser.add_argument(
            "--num-features",
            type=int,
            default=10,
            help="The number of superpixels to consider as features.",
        )
        return self

    # =========================================================================

    @property
    def output_feature(self) -> int | None:
        return None

    @property
    def model(self) -> Callable:
        model_cls = CNN if "cnn" in self.args.model.name.lower() else resnet18
        model_, state = eqx.nn.make_with_state(model_cls)(jax.random.PRNGKey(0))
        model_, state = eqx.tree_deserialise_leaves(self.args.model, (model_, state))
        model_ = eqx.nn.inference_mode(model_)

        @partial(jax.vmap, axis_name="batch")
        def model(x):
            y, _ = model_(x, state)
            return y

        return model

    @property
    def shap_variant(self) -> str:
        if "superfeature" in self.args.shap_variant:
            return self.args.shap_variant
        else:
            return f"superfeature-{self.args.shap_variant}"

    @property
    def data(self) -> Iterable[np.ndarray]:
        testset = torchvision.datasets.CIFAR10(
            ".datasets",
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
        return NumpyDataset(testset)

    @property
    def data_mean(self) -> np.ndarray:
        resource_dir = Path(__file__).parent / "resources"
        data_mean = np.load(resource_dir / "cifar10_train_mean.npy")
        return data_mean

    @property
    def _superpixels_path(self) -> Path:
        resources_dir = Path(__file__).parent / "resources"
        return resources_dir / "cifar10_superpixels_100"

    @property
    def _images(self) -> np.ndarray:
        return np.load(self._superpixels_path / "images.npz")

    @property
    def _masks(self) -> np.ndarray:
        return np.load(self._superpixels_path / "masks.npz")

    @property
    def sample(self) -> np.ndarray:
        num_features = self.args.num_features
        img = self._images[f"{num_features}"]
        img = jnp.asarray(img)
        return jnp.moveaxis(img, -1, 0)

    @property
    def masks(self) -> np.ndarray:
        num_features = self.args.num_features
        masks = self._masks[f"{num_features}"]
        masks = jnp.asarray(masks)
        return jnp.broadcast_to(masks, (masks.shape[0], 3, 32, 32))

    @property
    def base_mask(self) -> np.ndarray:
        if "superfeature" in self.shap_variant:
            num_features = self.args.num_features
            return jnp.ones(num_features, dtype=jnp.float32)
        else:
            return jnp.ones_like(self.sample)

    def out_file(self, local_output_dir: Path) -> Path:
        if self.args.out_file is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
            out_file = local_output_dir / (
                f"{self.args.model.stem}_{self.args.feature}_shap"
                f"_{self.method_name}_{self.args.shap_variant}_{timestamp}.csv"
            )
        else:
            out_file = Path(self.args.out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        return out_file
