# Copyright 2025 David Boetius
"""Utilities for parsing command line arguments for MNIST experiments."""

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
from .models import CNN


class NPDataset:
    def __init__(self, dataset: torch.utils.data.Dataset):
        self.dataset = dataset

    def __getitem__(self, index) -> np.ndarray:
        data = self.dataset.data[index].numpy()
        data = data.reshape(1, 28, 28) / 255.0
        return data

    def __iter__(self) -> Iterator[np.ndarray]:
        for index in range(len(self.dataset)):
            yield self[index]

    def __len__(self) -> int:
        return len(self.dataset)


class MNISTCmdArgs(CmdArgs):
    def __init__(self, *parser_args, **parser_kwargs):
        super().__init__(*parser_args, **parser_kwargs)

    def dataset_args(
        self, default_dataset: str = None, default_data_index: int = 0
    ) -> "MNISTCmdArgs":
        self.data_index_args(default_data_index)
        return self

    def segmentation_args(self) -> "MNISTCmdArgs":
        self.parser.add_argument(
            "--num-patches",
            type=str,
            default="28,28",
            help="The number of patches to consider as features.",
        )
        return self

    # =========================================================================

    @property
    def data(self) -> Iterable[np.ndarray]:
        testset = torchvision.datasets.MNIST(
            ".datasets",
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
        return NPDataset(testset)

    @property
    def model(self) -> Callable:
        model_, state = eqx.nn.make_with_state(CNN)(jax.random.PRNGKey(0))
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
    def num_patches(self) -> tuple[tuple[int, int], int]:
        num_patches = self.args.num_patches
        try:
            num_patches = int(num_patches)
            num_patches = (num_patches, num_patches)
        except ValueError:
            num_patches = tuple(int(x) for x in num_patches.split(","))
        assert num_patches[0] <= 28
        assert num_patches[1] <= 28
        total_patches = num_patches[0] * num_patches[1]
        num_patches = (1, *num_patches)
        return num_patches, total_patches

    @property
    def feature(self) -> tuple[int, ...]:
        in_feature = self.args.feature
        num_patches, _ = self.num_patches
        try:
            in_feature = int(in_feature)
        except ValueError:
            in_feature = tuple(int(x) for x in in_feature.split(","))
            in_feature = in_feature[0] * num_patches[0] + in_feature[1]
        return in_feature

    @property
    def masks(self) -> np.ndarray:
        # create patch masks
        num_patches, total_patches = self.num_patches

        img_size = (1, 28, 28)
        mask_idx = jnp.arange(total_patches).reshape(num_patches)
        mask_idx = jax.image.resize(mask_idx, img_size, "nearest")
        masks = jnp.arange(total_patches).reshape((total_patches, 1, 1, 1)) == mask_idx
        return masks

    def out_file(self, local_output_dir: Path) -> Path:
        if self.args.out_file is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
            out_file = (
                local_output_dir
                / f"{self.args.dataset}"
                / (
                    f"{self.args.model.stem}_{self.args.feature}_{self.args.output_feature}_shap"
                    f"_{self.method_name}_{self.args.shap_variant}_{timestamp}.csv"
                )
            )
        else:
            out_file = Path(self.args.out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        return out_file
