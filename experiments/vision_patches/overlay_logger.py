# Copyright 2025 David Boetius
from pathlib import Path
from typing import Iterable

import colour
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import skimage.color
from matplotlib import colormaps, colors
from matplotlib import pyplot as plt
from tqdm import tqdm

from shap_bounds.logger import Logger


def adapt_luminance(color, luminance_multiplier):
    color_xyz = skimage.color.rgb2xyz(color[..., :3])
    color_oklab = colour.XYZ_to_Oklab(color_xyz)
    color_oklab[..., 0] = color_oklab[..., 0] * luminance_multiplier
    color_xyz = colour.Oklab_to_XYZ(color_oklab)
    color_rgb = skimage.color.xyz2rgb(color_xyz)
    return np.concatenate([color_rgb, color[..., 3:]], axis=-1)


class VisionPatchesBoundsLogger(Logger):
    def __init__(
        self,
        sample: np.ndarray,
        num_patches: tuple[int, int, int],
        output_dir: Path,
        feature: int | None = None,
        show: bool = True,
        base_alpha: float = 0.7,
        dpi: int = 150,
        midpoint_scale_max_abs: float | None = None,
        range_scale_max_abs: float | None = None,
        separate_colorbar: bool = False,
        more_space: bool = False,
    ):
        self.sample = np.asarray(sample)
        self.num_patches = num_patches
        self.feature = feature
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir = self.output_dir / "bounds_images"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.show = show
        self.base_alpha = base_alpha
        self.dpi = dpi
        self._pad_inches = 0.15 if more_space else 0.0

        self.mask_idx = self._make_mask_idx()
        # self.cmap = self._load_colormap("vik").reversed()
        self.cmap = self._load_colormap("roma")
        self.mid_cmap = self.cmap
        self.fig = None
        self.ax = None
        self.iter_ax = None
        self.base_artist = None
        self.overlay_artist = None
        self.iter_text = None
        self.combined_colorbar_ax = None
        self.combined_colorbar_image = None
        self.combined_colorbar_size = 256
        self.mid_norm = None
        self.range_max_abs = None
        self.separate_colorbar = separate_colorbar
        self._colorbar_fixed = (
            midpoint_scale_max_abs is not None and range_scale_max_abs is not None
        )
        self._colorbar_saved = False
        self._combined_colorbar_rgba = None
        self._combined_colorbar_extent = None
        self.mid_scale_max_abs = (
            float(midpoint_scale_max_abs)
            if midpoint_scale_max_abs is not None and midpoint_scale_max_abs > 0
            else None
        )
        self.range_scale_max_abs = (
            float(range_scale_max_abs)
            if range_scale_max_abs is not None and range_scale_max_abs > 0
            else None
        )

    def _load_colormap(self, name: str):
        try:
            return colormaps[name]
        except KeyError:
            try:
                from cmcrameri import cm as cmc
            except ImportError as exc:
                raise ImportError(
                    f"Colormap '{name}' not found. Install cmcrameri to use it."
                ) from exc
            return getattr(cmc, name)

    def _make_mask_idx(self) -> np.ndarray:
        channels, height, width = self.sample.shape
        total_patches = int(np.prod(self.num_patches))
        mask_idx = jnp.arange(total_patches, dtype=jnp.int32).reshape(self.num_patches)
        mask_idx = jax.image.resize(mask_idx, (channels, height, width), "nearest")
        return np.asarray(mask_idx, dtype=np.int32)

    def _ensure_figure(self):
        if self.fig is not None:
            return
        if self.show:
            plt.ion()
        self.fig = plt.figure(figsize=(9.0, 6.5), dpi=self.dpi)
        outer = self.fig.add_gridspec(
            nrows=1, ncols=2, width_ratios=[4, 1], wspace=0.25, hspace=0.0
        )
        self.ax = self.fig.add_subplot(outer[0, 0])
        self.ax.set_axis_off()
        right = outer[0, 1].subgridspec(
            nrows=8,
            ncols=1,
            height_ratios=[1, 1, 1, 1, 0.8, 0.6, 0.4, 0.4],
            hspace=0.45,
        )
        self.combined_colorbar_ax = self.fig.add_subplot(right[:5, 0])
        base_image = self._base_image()
        if base_image.ndim == 2:
            self.base_artist = self.ax.imshow(base_image, cmap="gray", vmin=0, vmax=1)
        else:
            self.base_artist = self.ax.imshow(base_image, vmin=0, vmax=1)
        overlay = np.zeros((*base_image.shape[:2], 4), dtype=np.float32)
        self.overlay_artist = self.ax.imshow(overlay)
        self.iter_ax = self.fig.add_subplot(right[5:, 0])
        self.iter_ax.set_axis_off()
        self.iter_text = self.iter_ax.text(
            0.5, 0.5, "", ha="center", va="center", fontsize=11
        )
        self._init_colorbars()

    def _base_image(self) -> np.ndarray:
        if self.sample.ndim == 3 and self.sample.shape[0] == 1:
            base = self.sample[0]
        else:
            base = np.moveaxis(self.sample, 0, -1)
        return np.clip(base, 0.0, 1.0)

    def _init_colorbars(self):
        self.mid_norm = colors.Normalize(vmin=-1.0, vmax=1.0)
        if self.combined_colorbar_ax is None:
            return
        self.combined_colorbar_ax.set_xlabel("bounds range", labelpad=6)
        self.combined_colorbar_ax.set_ylabel("midpoint")
        self.combined_colorbar_image = self.combined_colorbar_ax.imshow(
            np.zeros((self.combined_colorbar_size, self.combined_colorbar_size, 4)),
            origin="lower",
        )

    def _prepare_bounds(self, bounds: tuple) -> tuple[np.ndarray, np.ndarray]:
        lbs, ubs = bounds
        mid = np.asarray((lbs + ubs) / 2).reshape(-1)
        ran = np.asarray((ubs - lbs) / 2).reshape(-1)

        total_patches = int(np.prod(self.num_patches))
        if mid.size == 1 and self.feature is not None and total_patches > 1:
            mid_full = np.zeros(total_patches, dtype=mid.dtype)
            ran_full = np.zeros(total_patches, dtype=ran.dtype)
            mid_full[self.feature] = mid.item()
            ran_full[self.feature] = ran.item()
            mid, ran = mid_full, ran_full
        return mid, ran

    def _overlay_rgba(self, mid: np.ndarray, ran: np.ndarray) -> np.ndarray:
        max_abs_mid = float(np.max(np.abs(mid))) if mid.size > 0 else 0.0
        if self.mid_scale_max_abs is None:
            self.mid_scale_max_abs = max_abs_mid if max_abs_mid > 0 else 1.0
        scale_max = self.mid_scale_max_abs
        if scale_max > 0:
            intensity = 1.0 - np.clip(ran / scale_max, 0.0, 1.0)
        else:
            intensity = np.zeros_like(mid, dtype=np.float32)
            scale_max = 1.0

        mid_map = mid[self.mask_idx]
        intensity_map = intensity[self.mask_idx]

        if mid_map.ndim == 3:
            mid_map = mid_map.mean(axis=0)
            intensity_map = intensity_map.mean(axis=0)

        self._update_mid_colorbar(max_abs_mid)
        mid_colors = self.mid_cmap(self.mid_norm(mid_map))
        overlay = adapt_luminance(mid_colors, np.clip(intensity_map, 0.0, 1.0))
        overlay[..., 3] = self.base_alpha
        return overlay

    def _update_mid_colorbar(self, max_abs_mid: float) -> None:
        if self.mid_scale_max_abs is not None:
            max_abs_mid = self.mid_scale_max_abs
        if max_abs_mid <= 0:
            max_abs_mid = 1.0
        if (
            self.mid_norm is None
            or self.mid_norm.vmin != -max_abs_mid
            or self.mid_norm.vmax != max_abs_mid
        ):
            self.mid_norm = colors.Normalize(vmin=-max_abs_mid, vmax=max_abs_mid)
        self._update_combined_colorbar(max_abs_mid)

    def _update_range_colorbar(self) -> None:
        if self.range_max_abs is None:
            if self.range_scale_max_abs is not None:
                self.range_max_abs = self.range_scale_max_abs
            else:
                if self.mid_scale_max_abs is None or self.mid_scale_max_abs <= 0:
                    self.mid_scale_max_abs = 1.0
                self.range_max_abs = self.mid_scale_max_abs
        mid_max_abs = 1.0
        if self.mid_norm is not None:
            mid_max_abs = max(abs(self.mid_norm.vmin), abs(self.mid_norm.vmax))
        self._update_combined_colorbar(mid_max_abs)

    def _update_combined_colorbar(self, max_abs_mid: float) -> None:
        if self.combined_colorbar_image is None:
            return
        if max_abs_mid <= 0:
            max_abs_mid = 1.0
        range_max_abs = self.range_max_abs if self.range_max_abs is not None else 1.0
        if range_max_abs <= 0:
            range_max_abs = 1.0
        size = self.combined_colorbar_size
        y_vals = np.linspace(-max_abs_mid, max_abs_mid, size)
        x_vals = np.linspace(0.0, range_max_abs, size)
        mid_colors = self.mid_cmap(self.mid_norm(y_vals))
        intensity = 1.0 - np.clip(x_vals / range_max_abs, 0.0, 1.0)
        mid_colors = np.tile(mid_colors[:, None, :3], (1, size, 1))
        intensity = np.tile(intensity[None, :], (size, 1))
        rgb = adapt_luminance(mid_colors, intensity)
        # rgb = mid_colors[:, None, :3] * intensity[None, :, None]
        alpha = np.ones((size, size, 1), dtype=rgb.dtype)
        image = np.concatenate([rgb, alpha], axis=-1)
        self.combined_colorbar_image.set_data(image)
        self._combined_colorbar_rgba = image
        self._combined_colorbar_extent = (0.0, range_max_abs, -max_abs_mid, max_abs_mid)
        self.combined_colorbar_image.set_extent(self._combined_colorbar_extent)
        self.combined_colorbar_ax.set_xlim(0.0, range_max_abs)
        self.combined_colorbar_ax.set_ylim(-max_abs_mid, max_abs_mid)

    def _save_main_image(self, out_file: Path) -> None:
        if self.fig is None or self.ax is None:
            return
        bbox = self.ax.get_window_extent().transformed(
            self.fig.dpi_scale_trans.inverted()
        )
        self.fig.savefig(
            out_file, dpi=self.dpi, bbox_inches=bbox, pad_inches=self._pad_inches
        )

    def _save_colorbar_assets(self, base_path: Path) -> None:
        if (
            self._combined_colorbar_rgba is None
            or self._combined_colorbar_extent is None
        ):
            return
        if self._colorbar_fixed and self._colorbar_saved:
            return
        fig, ax = plt.subplots(figsize=(4.5, 4.0), dpi=self.dpi)
        ax.imshow(
            self._combined_colorbar_rgba,
            origin="lower",
            extent=self._combined_colorbar_extent,
            aspect="auto",
        )
        ax.set_xlabel("bounds range")
        ax.set_ylabel("midpoint")
        fig.tight_layout()
        fig.savefig(base_path, dpi=self.dpi, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        xmin, xmax, ymin, ymax = self._combined_colorbar_extent
        scale_path = base_path.with_suffix(".txt")
        scale_path.write_text(f"ymin={ymin}\nymax={ymax}\nxmin={xmin}\nxmax={xmax}\n")
        if self._colorbar_fixed:
            self._colorbar_saved = True

    def log_config(self, config_name: str, config1: dict | None = None, **config2):
        return None

    def log_stats(
        self,
        function_name: str,
        stats1: dict | None = None,
        temporary: bool = False,
        **stats2,
    ):
        pass

    def log_iter_stats(
        self, function_name: str, i: int, stats1: dict | None = None, **stats2
    ):
        pass

    def log_bounds(
        self,
        function_name: str,
        i: int,
        bounds: tuple,
        name: str = "φ",
        runtime: float | None = None,
    ):
        self._ensure_figure()
        mid, ran = self._prepare_bounds(bounds)
        self._update_range_colorbar()
        overlay = self._overlay_rgba(mid, ran)
        self.overlay_artist.set_data(overlay)
        if self.iter_text is not None:
            self.iter_text.set_text(f"t={i}")
        self.fig.canvas.draw_idle()
        if self.show:
            self.fig.canvas.flush_events()
            plt.pause(0.001)
        out_file = self.image_dir / f"{function_name}_bounds_{i:04d}.png"
        if self.separate_colorbar:
            self._save_main_image(out_file)
            if self._colorbar_fixed:
                colorbar_file = self.image_dir / f"{function_name}_colormap.png"
            else:
                colorbar_file = self.image_dir / f"{function_name}_colormap_{i:04d}.png"
            self._save_colorbar_assets(colorbar_file)
        else:
            self.fig.savefig(
                out_file,
                dpi=self.dpi,
                bbox_inches="tight",
                pad_inches=self._pad_inches,
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.fig is not None:
            plt.close(self.fig)


def _bounds_from_frame(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(frame.columns, pd.MultiIndex):
        lbs = frame.xs("lb", level=1, axis=1)
        ubs = frame.xs("ub", level=1, axis=1)
        return lbs.to_numpy(), ubs.to_numpy()
    if "lb" in frame.columns and "ub" in frame.columns:
        return frame[["lb"]].to_numpy(), frame[["ub"]].to_numpy()
    raise ValueError("Bounds file missing 'lb'/'ub' columns.")


def _iter_bounds(
    frame: pd.DataFrame,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    lbs, ubs = _bounds_from_frame(frame)
    for lb, ub in zip(lbs, ubs, strict=False):
        yield lb, ub


if __name__ == "__main__":
    from .argument_parser import VisionPatchesCmdArgs

    cmd_args = (
        VisionPatchesCmdArgs()
        .model_args()
        .dataset_args()
        .segmentation_args()
        .feature_args()
        .shap_variant_args()
    )
    cmd_args.parser.add_argument(
        "--bounds",
        type=Path,
        required=True,
        help="Path to a bounds .feather file.",
    )
    cmd_args.parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for replayed bounds images.",
    )
    cmd_args.parser.add_argument("--no-show", action="store_true")
    cmd_args.parser.add_argument("--dpi", type=int, default=150)
    cmd_args.parser.add_argument(
        "--separate-colormap",
        action="store_true",
        help="Save overlay images and colormap images as separate files.",
    )
    cmd_args.parser.add_argument(
        "--more-space",
        action="store_true",
        help="Add a small margin around saved images.",
    )
    cmd_args.parser.add_argument("--speed", type=int, default=1)
    args = cmd_args.parse_args()

    bounds_path = args.args.bounds
    bounds_df = pd.read_feather(bounds_path)
    lb_array, ub_array = _bounds_from_frame(bounds_df)
    mids = (lb_array + ub_array) / 2
    mid_scale = float(np.max(np.abs(mids))) * 1.0
    mid_final = (lb_array[-1] + ub_array[-1]) / 2
    range_scale = float(np.max(np.abs(mid_final)))

    sample = args.sample
    num_patches = args.num_patches[0]
    out_dir = args.args.out or bounds_path.parent / "replay"

    logger = VisionPatchesBoundsLogger(
        sample,
        num_patches,
        out_dir,
        feature=args.feature,
        show=not args.args.no_show,
        dpi=args.args.dpi,
        midpoint_scale_max_abs=mid_scale,
        range_scale_max_abs=range_scale,
        separate_colorbar=args.args.separate_colormap,
        more_space=args.args.more_space,
    )

    speed = args.args.speed
    for i, (lb, ub) in tqdm(enumerate(_iter_bounds(bounds_df)), total=len(bounds_df)):
        if i % speed == 0:
            logger.log_bounds("replay", i, (lb, ub))
    if i % speed != 0:
        logger.log_bounds("replay", i, (lb, ub))
