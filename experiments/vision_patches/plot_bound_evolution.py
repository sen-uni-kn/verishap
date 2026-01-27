# Copyright 2025 David Boetius
"""Plot bound evolution over time for selected features from a bounds file."""

import argparse
from pathlib import Path

import colour
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.color
import yaml
from matplotlib import colormaps, colors
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


def adapt_luminance(color, luminance_multiplier):
    """Adapt luminance of colors using Oklab color space.

    Args:
        color: RGB(A) colors, shape (..., 3) or (..., 4)
        luminance_multiplier: Multiplier for luminance channel, shape (...)

    Returns:
        RGB(A) colors with adjusted luminance
    """
    color_xyz = skimage.color.rgb2xyz(color[..., :3])
    color_oklab = colour.XYZ_to_Oklab(color_xyz)
    color_oklab[..., 0] = color_oklab[..., 0] * luminance_multiplier
    color_xyz = colour.Oklab_to_XYZ(color_oklab)
    color_rgb = skimage.color.xyz2rgb(color_xyz)
    return np.concatenate([color_rgb, color[..., 3:]], axis=-1)


def load_colormap(name: str):
    """Load a colormap by name, trying cmcrameri if not in matplotlib."""
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


def load_bounds_data(bounds_path: Path) -> tuple[pd.DataFrame, int]:
    """Load bounds data from feather file and determine number of features."""
    bounds_df = pd.read_feather(bounds_path)

    # Check if it's the bounds file (has feature columns) or iter_stats file
    if isinstance(bounds_df.columns, pd.MultiIndex):
        # Multi-index columns like ('0', 'lb'), ('0', 'ub')
        num_features = len(
            [
                col
                for col in bounds_df.columns.get_level_values(0).unique()
                if col != "runtime"
            ]
        )
    elif any(col[0].isdigit() for col in bounds_df.columns if isinstance(col, tuple)):
        # Columns are tuples like ('0', 'lb')
        num_features = len(
            [
                col
                for col in bounds_df.columns
                if isinstance(col, tuple) and col[1] == "lb"
            ]
        )
    else:
        # Try to infer from column names
        feature_cols = [
            col for col in bounds_df.columns if isinstance(col, tuple) and len(col) == 2
        ]
        if feature_cols:
            num_features = len([col for col in feature_cols if col[1] == "lb"])
        else:
            raise ValueError(
                f"Cannot determine feature structure from file {bounds_path}. "
                "Make sure this is a bounds file (not iter_stats)."
            )

    return bounds_df, num_features


def load_model_output(bounds_path: Path) -> float | None:
    """Load model output from info.yaml in the same directory."""
    info_path = bounds_path.parent / "info.yaml"
    if not info_path.exists():
        print(f"Warning: {info_path} not found, cannot compute 10% threshold marker")
        return None

    with info_path.open("r") as f:
        info = yaml.safe_load(f)

    model_output = info.get("config", {}).get("further_stats", {}).get("model_output")
    if model_output is None:
        print("Warning: model_output not found in info.yaml")
        return None

    return float(model_output)


def compute_line_colors(
    mids: np.ndarray,
    ranges: np.ndarray,
    scale_max: float,
    cmap,
    norm,
) -> np.ndarray:
    """Compute colors for line segments based on midpoint and half-width.

    Args:
        mids: Midpoint values, shape (n,)
        ranges: Half-width values, shape (n,)
        scale_max: Maximum absolute value for normalization
        cmap: Colormap to use
        norm: Normalization for the colormap

    Returns:
        RGBA colors, shape (n, 4)
    """
    # Map midpoints to colors
    mid_colors = cmap(norm(mids))

    # Compute intensity based on ranges (tighter bounds = brighter)
    if scale_max > 0:
        intensity = 1.0 - np.clip(ranges / scale_max, 0.0, 1.0)
    else:
        intensity = np.ones_like(ranges)

    # Adjust luminance
    colors_adjusted = adapt_luminance(mid_colors, intensity)

    return colors_adjusted


def plot_bound_evolution(
    bounds_df: pd.DataFrame,
    features: list[int],
    model_output: float | None = None,
    output_path: Path | None = None,
    export_path: Path | None = None,
    use_runtime: bool = False,
    figsize: tuple[float, float] = (10, 4),
    dpi: int = 150,
):
    """Plot bound evolution for specified features.

    Args:
        bounds_df: DataFrame with bounds data (multi-index columns)
        features: List of feature indices to plot
        model_output: Model output value for computing 10% threshold marker
        output_path: Optional path to save the plot
        export_path: Optional path to save the plot data as TSV
        use_runtime: If True, use runtime on x-axis; otherwise use iteration
        figsize: Figure size in inches
        dpi: DPI for the figure
    """
    # Load roma colormap
    cmap = load_colormap("roma")

    # Create figure with single subplot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Get x-axis values (runtime or iteration)
    if use_runtime and "runtime" in bounds_df.columns:
        x_values = bounds_df["runtime"].values
        x_label = "Runtime (s)"
    else:
        x_values = np.arange(len(bounds_df))
        x_label = "Iteration"

    # Track data range and compute 10% threshold marker
    all_lbs = []
    all_ubs = []
    all_mids = []
    all_ranges = []
    all_final_mids = []
    threshold_x = None

    # First pass: collect all data to determine scale_max
    for feat_idx in features:
        lb_col = (str(feat_idx), "lb")
        ub_col = (str(feat_idx), "ub")

        if lb_col not in bounds_df.columns or ub_col not in bounds_df.columns:
            print(f"Warning: Feature {feat_idx} not found in data, skipping.")
            continue

        lbs = bounds_df[lb_col].values
        ubs = bounds_df[ub_col].values
        mids = (lbs + ubs) / 2
        ranges = (ubs - lbs) / 2

        all_lbs.append(lbs)
        all_ubs.append(ubs)
        all_mids.append(mids)
        all_ranges.append(ranges)
        all_final_mids.append(mids[-1])

    if not all_mids:
        print("No valid features to plot")
        return

    # Compute scale_max as max absolute midpoint across all features and iterations
    all_mids_concat = np.concatenate(all_mids)
    scale_max = (
        float(np.max(np.abs(all_mids_concat))) if all_mids_concat.size > 0 else 1.0
    )
    if scale_max <= 0:
        scale_max = 1.0

    # Create normalization for colormap (symmetric around 0)
    norm = colors.Normalize(vmin=-scale_max, vmax=scale_max)

    # Second pass: plot with computed colors
    legend_handles = []
    legend_labels = []

    # Store color data for export
    export_data = {}
    if export_path is not None:
        export_data["Iteration"] = x_values

    for i, feat_idx in enumerate(features):
        if i >= len(all_lbs):
            continue

        lbs = all_lbs[i]
        ubs = all_ubs[i]
        mids = all_mids[i]
        ranges = all_ranges[i]

        # Find when this feature reaches 10% threshold
        if model_output is not None and threshold_x is None:
            threshold = 0.1 * abs(model_output)
            reached_threshold = ranges <= threshold
            if reached_threshold.any():
                threshold_idx = np.where(reached_threshold)[0][0]
                threshold_x = x_values[threshold_idx]

        # Compute colors for each point
        line_colors = compute_line_colors(mids, ranges, scale_max, cmap, norm)

        # Store data for export
        if export_path is not None:
            export_data[f"lb_{feat_idx}"] = lbs
            export_data[f"ub_{feat_idx}"] = ubs
            # Convert RGBA colors to rgb=r,g,b format (only RGB, ignore alpha)
            lb_colors = [f"rgb={c[0]:.6f},{c[1]:.6f},{c[2]:.6f}" for c in line_colors]
            ub_colors = [f"rgb={c[0]:.6f},{c[1]:.6f},{c[2]:.6f}" for c in line_colors]
            export_data[f"lb_{feat_idx}_color"] = lb_colors
            export_data[f"ub_{feat_idx}_color"] = ub_colors

        # Create line segments for lower bound
        points_lb = np.array([x_values, lbs]).T.reshape(-1, 1, 2)
        segments_lb = np.concatenate([points_lb[:-1], points_lb[1:]], axis=1)
        lc_lb = LineCollection(segments_lb, colors=line_colors[:-1], linewidths=1.5)
        ax.add_collection(lc_lb)

        # Create line segments for upper bound
        points_ub = np.array([x_values, ubs]).T.reshape(-1, 1, 2)
        segments_ub = np.concatenate([points_ub[:-1], points_ub[1:]], axis=1)
        lc_ub = LineCollection(segments_ub, colors=line_colors[:-1], linewidths=1.5)
        ax.add_collection(lc_ub)

        # Fill between with similar coloring (use average color for each segment)
        for j in range(len(x_values) - 1):
            fill_color = line_colors[j].copy()
            fill_color[3] = 0.3  # Set alpha for fill
            ax.fill_between(
                x_values[j : j + 2],
                lbs[j : j + 2],
                ubs[j : j + 2],
                color=fill_color,
                linewidth=0,
            )

        # Create legend entry with the color of the final (rightmost) bound
        final_color = line_colors[-1]
        legend_handle = Line2D([0], [0], color=final_color, linewidth=2)
        legend_handles.append(legend_handle)
        legend_labels.append(f"Feature {feat_idx}")

    # Compute when ANY feature reaches 10% threshold
    if model_output is not None and len(all_ranges) > 0:
        max_ranges = np.max(np.array(all_ranges), axis=0)
        threshold = 0.1 * abs(model_output)
        reached_threshold = max_ranges <= threshold
        if reached_threshold.any():
            threshold_idx = np.where(reached_threshold)[0][0]
            threshold_x = x_values[threshold_idx]

    # Add threshold marker if found
    if threshold_x is not None:
        threshold_line = ax.axvline(
            threshold_x, color="red", linestyle="--", linewidth=2, alpha=0.7
        )
        legend_handles.append(threshold_line)
        legend_labels.append("10% threshold")

    # Calculate y-axis limits: 10 * difference of final bound midpoints
    if all_final_mids:
        final_mid_min = np.min(all_final_mids)
        final_mid_max = np.max(all_final_mids)
        final_mid_center = (final_mid_min + final_mid_max) / 2
        final_mid_diff = final_mid_max - final_mid_min

        # Set range to 10 * the difference of final midpoints
        y_range = 10 * final_mid_diff if final_mid_diff > 0 else 1.0
        y_min = final_mid_center - y_range / 2
        y_max = final_mid_center + y_range / 2
        ax.set_ylim(y_min, y_max)

    # Set x-axis limits to show full range from iteration 0 to end (after autoscale_view)
    if len(x_values) > 0:
        # Add a tiny margin to ensure endpoints are visible
        x_margin = (x_values[-1] - x_values[0]) * 0.01
        ax.set_xlim(x_values[0] - x_margin, x_values[-1] + x_margin)

    # Formatting
    ax.set_xlabel(x_label)
    ax.set_ylabel("SHAP Value Bounds")
    ax.set_title("Bound Evolution Over Time")
    ax.grid(True, alpha=0.3)

    # Create legend with custom handles
    if legend_handles:
        ax.legend(legend_handles, legend_labels, fontsize=9, loc="best")

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close()

    # Export data to TSV if requested
    if export_path is not None and export_data:
        export_df = pd.DataFrame(export_data)
        export_df.to_csv(export_path, sep="\t", index=False)
        print(f"Saved plot data to {export_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot bound evolution over time for selected features."
    )
    parser.add_argument(
        "bounds_file",
        type=Path,
        help="Path to bounds feather file (e.g., multi_shap_bab_bounds.feather)",
    )
    parser.add_argument(
        "--features",
        type=int,
        nargs="+",
        required=True,
        help="Feature indices to plot (e.g., --features 7 2 19)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file path for the plot (if not specified, will show interactively)",
    )
    parser.add_argument(
        "--use-runtime",
        action="store_true",
        help="Use runtime instead of iteration number on x-axis",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[10, 4],
        help="Figure size in inches (width height)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for the output figure",
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help="Export plot data to TSV file with bounds and colors",
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading bounds from {args.bounds_file}")
    bounds_df, num_features = load_bounds_data(args.bounds_file)
    print(f"Found {num_features} features with {len(bounds_df)} iterations")

    # Load model output for threshold marker
    model_output = load_model_output(args.bounds_file)
    if model_output is not None:
        print(f"Model output: {model_output:.4f}")

    # Validate feature indices
    invalid_features = [f for f in args.features if f < 0 or f >= num_features]
    if invalid_features:
        raise ValueError(
            f"Invalid feature indices: {invalid_features}. "
            f"Valid range is 0 to {num_features - 1}."
        )

    # Plot
    plot_bound_evolution(
        bounds_df,
        args.features,
        model_output=model_output,
        output_path=args.output,
        export_path=args.export,
        use_runtime=args.use_runtime,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
