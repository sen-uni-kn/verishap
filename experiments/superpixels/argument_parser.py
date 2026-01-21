# Copyright 2025 David Boetius
"""Utilities for parsing command line arguments for CIFAR10 experiments."""

from datetime import datetime, timezone
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ..argument_parser import CmdArgs


class SuperpixelsCmdArgs(CmdArgs):
    def __init__(self, *parser_args, **parser_kwargs):
        super().__init__(*parser_args, **parser_kwargs)

    def model_args(self, default_model: Path | None = None):
        if default_model is None:
            resoure_dir = Path(__file__).parent.parent / "resources"
            default_model = resoure_dir / "cifar10-cnn.eqx"
        return super().model_args(default_model)

    def segmentation_args(self):
        self.parser.add_argument(
            "--num-features",
            type=int,
            default=10,
            help="The number of superpixels to consider as features.",
        )
        return self

    def logger_args(self) -> "SuperpixelsCmdArgs":
        super().logger_args()
        self.parser.add_argument(
            "--overlay-logger",
            action="store_true",
            help="Enable overlay logging for superpixels bounds.",
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
    def _superpixels_path(self) -> Path:
        if self.dataset.lower() == "cifar10":
            resources_dir = Path(__file__).parent / "resources"
            return resources_dir / "cifar10_superpixels_100"
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

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

    @property
    def overlay_logger(self) -> bool:
        return self.args.overlay_logger

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
