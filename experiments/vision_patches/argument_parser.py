# Copyright 2025 David Boetius
"""Utilities for parsing command line arguments for MNIST experiments."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ..argument_parser import CmdArgs


class VisionPatchesCmdArgs(CmdArgs):
    def __init__(self, *parser_args, **parser_kwargs):
        super().__init__(*parser_args, **parser_kwargs)

    def model_args(self, default_model: Path | None = None) -> "VisionPatchesCmdArgs":
        if default_model is None:
            resoure_dir = Path(__file__).parent.parent / "resources"
            default_model = resoure_dir / "mnist-cnn.eqx"
        return super().model_args(default_model)

    def segmentation_args(self) -> "VisionPatchesCmdArgs":
        self.parser.add_argument(
            "--num-patches",
            type=str,
            default="5,5",
            help="The number of patches to consider as features.",
        )
        return self

    # =========================================================================

    @property
    def img_shape(self) -> tuple[int, int, int]:
        return self.sample.shape

    @property
    def shap_variant(self) -> str:
        if "superfeature" in self.args.shap_variant:
            return self.args.shap_variant
        else:
            return f"superfeature-{self.args.shap_variant}"

    @property
    def num_patches(self) -> tuple[tuple[int, int, int], int]:
        img_shape = self.img_shape
        channels = img_shape[0]
        num_patches = self.args.num_patches
        try:
            num_patches = int(num_patches)
            # separate all channels
            num_patches = (channels, num_patches, num_patches)
        except ValueError:
            num_patches = tuple(int(x) for x in num_patches.split(","))
            if len(num_patches) == 2:
                num_patches = (channels, *num_patches)
        assert num_patches[0] <= img_shape[0]
        assert num_patches[1] <= img_shape[1]
        assert num_patches[2] <= img_shape[2]
        total_patches = num_patches[0] * num_patches[1] * num_patches[2]
        return num_patches, total_patches

    @property
    def feature(self) -> tuple[int, ...]:
        in_feature = self.args.feature
        num_patches, _ = self.num_patches
        if in_feature is None:
            return None
        try:
            in_feature = int(in_feature)
        except TypeError:
            in_feature = tuple(int(x) for x in in_feature.split(","))
            in_feature = in_feature[0] * num_patches[0] + in_feature[1]
        return in_feature

    @property
    def masks(self) -> np.ndarray:
        # create patch masks
        num_patches, total_patches = self.num_patches

        mask_idx = jnp.arange(total_patches).reshape(num_patches)
        mask_idx = jax.image.resize(mask_idx, self.img_shape, "nearest")
        masks = jnp.arange(total_patches).reshape((total_patches, 1, 1, 1)) == mask_idx
        return masks

    @property
    def base_mask(self) -> np.ndarray:
        if "superfeature" in self.shap_variant:
            _, total_patches = self.num_patches
            return jnp.ones(total_patches, dtype=jnp.float32)
        else:
            return jnp.ones_like(self.sample)
