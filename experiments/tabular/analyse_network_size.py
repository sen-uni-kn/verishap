# Copyright 2025 David Boetius
"""Analysis script for network_size_experiment.sh outputs.

Analyzes how runtime and bound quality scale with network size (width × depth).
"""

import argparse
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import yaml

VERBOSE = True
TIMEOUT = 600  # Runtime threshold in seconds - values above this indicate a timeout


def _parse_network_size(network_name: str) -> tuple[int | None, int | None]:
    """Extract width and depth from network name like 'mushroom-mlp-1024x1'."""
    match = re.search(r"-(\d+)x(\d+)$", network_name)
    if match:
        width = int(match.group(1))
        depth = int(match.group(2))
        return width, depth
    return None, None


def _compute_num_params(
    num_features: int, width: int | None, depth: int | None, output_dim: int = 1
) -> int | None:
    """Compute number of parameters for an MLP.

    Architecture: input -> [hidden layers] -> output
    - First hidden layer: num_features * width + width (weights + biases)
    - Additional hidden layers: (depth-1) * (width * width + width)
    - Output layer: width * output_dim + output_dim
    """
    if width is None or depth is None or num_features is None:
        return None

    # First hidden layer
    params = num_features * width + width

    # Additional hidden layers (depth - 1 more)
    if depth > 1:
        params += (depth - 1) * (width * width + width)

    # Output layer
    params += width * output_dim + output_dim

    return params


def _get_runtime_at(condition: np.ndarray, iter_times: np.ndarray) -> float | None:
    """Get the runtime when a condition is first satisfied."""
    idx = np.where(condition)[0]
    if len(idx) == 0:
        return None
    return iter_times[idx[0]]


def _load_run(info_path: Path) -> dict | None:
    """Load a single run from an info.yaml file."""
    if VERBOSE:
        print("Loading", info_path)

    with info_path.open("r") as f:
        info = yaml.safe_load(f)

    # Check if info file has overall entry
    has_overall = "overall" in info and info["overall"] is not None

    runtime = info.get("overall", {}).get("runtime", None)
    timeout = info.get("overall", {}).get("timeout", False)
    max_iters = info.get("overall", {}).get("max_iters", False)
    iterations = info.get("multi_shap_bab", {}).get("iterations", None)
    total_branches = info.get("multi_shap_bab", {}).get("total_branches", None)
    total_tight_bounds = info.get("multi_shap_bab", {}).get("total_tight_bounds", None)

    config = info.get("config", {})
    multi_shap_bab_config = config.get("multi_shap_bab", {})
    features = multi_shap_bab_config.get("features", [])
    num_features = len(features)

    # Check if bounds file exists
    bounds_path = info_path.parent / "multi_shap_bab_bounds.feather"
    has_bounds_file = bounds_path.exists()

    # Track success: must have overall entry AND bounds file
    success = has_overall and has_bounds_file

    # Initialize bound timing metrics
    time_to_10pct = None
    time_to_1pct = None
    final_bounds_pct = None

    # Load bounds if available
    if has_bounds_file:
        model_output = config.get("further_stats", {}).get("model_output")
        if model_output is not None:
            bounds = pd.read_feather(bounds_path)
            iter_times = bounds["runtime"]
            lbs = np.array([bounds[(f"{i}", "lb")] for i in range(num_features)]).T
            ubs = np.array([bounds[(f"{i}", "ub")] for i in range(num_features)]).T

            # Compute normalized ranges relative to model output
            ref_vals = np.abs(model_output)
            ranges_norm = ((ubs - lbs) / 2) / ref_vals
            max_norm_ranges = ranges_norm.max(axis=-1)

            time_to_10pct = _get_runtime_at(max_norm_ranges <= 0.1, iter_times)
            time_to_1pct = _get_runtime_at(max_norm_ranges <= 0.01, iter_times)

            # Get final bounds percentage (half-width as percentage of model output)
            if len(max_norm_ranges) > 0:
                final_bounds_pct = max_norm_ranges[-1] * 100  # Convert to percentage

    return {
        "runtime": runtime,
        "timeout": timeout,
        "max_iters": max_iters,
        "iterations": iterations,
        "total_branches": total_branches,
        "total_tight_bounds": total_tight_bounds,
        "num_features": num_features,
        "success": success,
        "time_to_10pct": time_to_10pct,
        "time_to_1pct": time_to_1pct,
        "final_bounds_pct": final_bounds_pct,
    }


def _parse_path(info_path: Path) -> dict | None:
    """Extract network and repetition from path.

    Expected structure: .../BaB/network_name/repeatition_i/info.yaml
    """
    parts = info_path.parts
    if len(parts) < 4:
        return None

    repetition_dir = parts[-2]
    network = parts[-3]
    category = parts[-4]

    # Skip warmup runs
    if category == "warmup":
        return None

    # Parse repetition number
    if not repetition_dir.startswith("repeatition_"):
        return None
    try:
        repetition = int(repetition_dir.split("_")[1])
    except (IndexError, ValueError):
        return None

    # Parse network size from name
    width, depth = _parse_network_size(network)

    return {
        "network": network,
        "width": width,
        "depth": depth,
        "repetition": repetition,
    }


def _iter_runs(data_dir: Path) -> Iterable[dict]:
    """Iterate over all runs in a data directory."""
    for info_path in data_dir.rglob("info.yaml"):
        path_info = _parse_path(info_path)
        if path_info is None:
            continue

        run = _load_run(info_path)
        if run is not None:
            run.update(path_info)
            # Compute number of parameters from network architecture
            run["num_params"] = _compute_num_params(
                run["num_features"], run["width"], run["depth"]
            )
            yield run


def _load_data_dir(data_dir: Path) -> pd.DataFrame:
    """Load all runs from a data directory and compute statistics."""
    runs = list(_iter_runs(data_dir))

    if not runs:
        raise SystemExit(f"No runs found in {data_dir}")

    df = pd.DataFrame(runs)

    # Group by network and compute median runtime
    grouped = df.groupby("network").agg(
        {
            "runtime": "median",
            "timeout": "any",
            "max_iters": "any",
            "iterations": "median",
            "total_branches": "median",
            "total_tight_bounds": "median",
            "num_features": "first",
            "width": "first",
            "depth": "first",
            "num_params": "first",
            "success": "all",  # All repetitions must succeed
            "time_to_10pct": "median",
            "time_to_1pct": "median",
            "final_bounds_pct": "median",
        }
    )

    # Also compute runtime std for error bars
    runtime_std = df.groupby("network")["runtime"].std()
    grouped["runtime_std"] = runtime_std

    grouped = grouped.reset_index()

    # Save the full grouped data
    grouped.to_csv(data_dir / "network_size_runtimes.csv", index=False)

    return grouped


def _format_table(df: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    """Format a DataFrame as a simple aligned table."""
    # Filter to requested columns
    df_display = df[columns].copy()
    df_display.columns = headers

    # Calculate column widths
    col_widths = []
    for col in df_display.columns:
        header_width = len(str(col))
        # For #Params, calculate width based on comma-formatted numbers
        if col == "#Params":
            data_width = df_display[col].apply(
                lambda x: len(f"{int(x):,}") if pd.notna(x) else 2
            ).max()
        else:
            data_width = df_display[col].astype(str).str.len().max()
        col_widths.append(max(header_width, data_width))

    # Build header
    header_parts = []
    for col, width in zip(df_display.columns, col_widths):
        header_parts.append(str(col).rjust(width))
    lines = [" | ".join(header_parts)]
    lines.append("-" * len(lines[0]))

    # Build rows
    for _, row in df_display.iterrows():
        row_parts = []
        for col_name, (val, width) in zip(df_display.columns, zip(row, col_widths)):
            if pd.isna(val):
                row_parts.append("--".rjust(width))
            elif isinstance(val, float):
                # Use integer formatting for Width/Depth/Iterations columns
                if col_name in ["Width", "Depth", "Iterations"]:
                    row_parts.append(f"{int(val)}".rjust(width))
                elif col_name == "#Params":
                    row_parts.append(f"{int(val):,}".rjust(width))
                else:
                    row_parts.append(f"{val:.2f}".rjust(width))
            else:
                row_parts.append(str(val).rjust(width))
        lines.append(" | ".join(row_parts))

    return "\n".join(lines)


def main(data_dirs: Sequence[Path], quiet: bool = False, latex: bool = False) -> None:
    global VERBOSE
    VERBOSE = not quiet

    # Load data from all directories
    all_dfs = []
    for data_dir in data_dirs:
        df = _load_data_dir(data_dir)
        all_dfs.append(df)

    # Combine data from multiple directories
    if len(all_dfs) == 1:
        combined = all_dfs[0]
    else:
        concat = pd.concat(all_dfs)
        combined = (
            concat.groupby("network")
            .agg(
                {
                    "runtime": "median",
                    "runtime_std": "mean",
                    "timeout": "any",
                    "max_iters": "any",
                    "iterations": "median",
                    "total_branches": "median",
                    "total_tight_bounds": "median",
                    "num_features": "first",
                    "width": "first",
                    "depth": "first",
                    "num_params": "first",
                    "success": "all",
                    "time_to_10pct": "median",
                    "time_to_1pct": "median",
                    "final_bounds_pct": "median",
                }
            )
            .reset_index()
        )

    # Sort by width, then depth
    combined = combined.sort_values(["depth", "width"])

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.precision", 2)
    pd.set_option("display.width", 200)

    # Separate by depth variation
    depth_1 = combined[combined["depth"] == 1].copy()
    depth_varying = combined[combined["depth"] > 1].copy()

    # Print summary statistics
    print(f"\n{'=' * 80}")
    print(" NETWORK SIZE EXPERIMENT ANALYSIS")
    print(f"{'=' * 80}\n")

    # Success summary
    total_runs = len(combined)
    successful_runs = combined["success"].sum()
    print(f"Total network configurations: {total_runs}")
    print(
        f"Successful runs: {successful_runs} ({100 * successful_runs / total_runs:.1f}%)"
    )
    print()

    # Width scaling (depth=1)
    if not depth_1.empty:
        print(f"\n{'-' * 80}")
        print(" WIDTH SCALING (depth=1)")
        print(f"{'-' * 80}\n")

        # Replace timeout values with NaN for display
        depth_1_display = depth_1.copy()
        depth_1_display.loc[depth_1_display["timeout"], "runtime"] = np.nan

        if latex:
            print("% Width Scaling Results (depth=1)")
            print(r"\begin{tabular}{rrrrrr}")
            print(r"\toprule")
            print(
                r"\textbf{Width} & \textbf{\#Params} & \textbf{Runtime (s)} & \textbf{Iterations} & "
                r"\textbf{Time to 10\%} & \textbf{Final \%} \\"
            )
            print(r"\midrule")
            for _, row in depth_1_display.iterrows():
                width = int(row["width"])
                num_params = (
                    f"{int(row['num_params']):,}" if pd.notna(row["num_params"]) else "--"
                )
                runtime = f"{row['runtime']:.2f}" if pd.notna(row["runtime"]) else "--"
                iterations = (
                    f"{int(row['iterations'])}" if pd.notna(row["iterations"]) else "--"
                )
                time_10 = (
                    f"{row['time_to_10pct']:.2f}"
                    if pd.notna(row["time_to_10pct"])
                    else "--"
                )
                final_pct = (
                    f"{row['final_bounds_pct']:.2f}"
                    if pd.notna(row["final_bounds_pct"])
                    else "--"
                )
                print(
                    f"{width} & {num_params} & {runtime} & {iterations} & {time_10} & {final_pct} \\\\"
                )
            print(r"\bottomrule")
            print(r"\end{tabular}")
        else:
            print(
                _format_table(
                    depth_1_display,
                    [
                        "width",
                        "num_params",
                        "runtime",
                        "iterations",
                        "time_to_10pct",
                        "final_bounds_pct",
                    ],
                    ["Width", "#Params", "Runtime (s)", "Iterations", "Time to 10%", "Final %"],
                )
            )
        print()

    # Depth scaling (width=1024)
    if not depth_varying.empty:
        print(f"\n{'-' * 80}")
        print(" DEPTH SCALING (width=1024)")
        print(f"{'-' * 80}\n")

        # Replace timeout values with NaN for display
        depth_varying_display = depth_varying.copy()
        depth_varying_display.loc[depth_varying_display["timeout"], "runtime"] = np.nan

        if latex:
            print("% Depth Scaling Results (width=1024)")
            print(r"\begin{tabular}{rrrrrr}")
            print(r"\toprule")
            print(
                r"\textbf{Depth} & \textbf{\#Params} & \textbf{Runtime (s)} & \textbf{Iterations} & "
                r"\textbf{Time to 10\%} & \textbf{Final \%} \\"
            )
            print(r"\midrule")
            for _, row in depth_varying_display.iterrows():
                depth = int(row["depth"])
                num_params = (
                    f"{int(row['num_params']):,}" if pd.notna(row["num_params"]) else "--"
                )
                runtime = f"{row['runtime']:.2f}" if pd.notna(row["runtime"]) else "--"
                iterations = (
                    f"{int(row['iterations'])}" if pd.notna(row["iterations"]) else "--"
                )
                time_10 = (
                    f"{row['time_to_10pct']:.2f}"
                    if pd.notna(row["time_to_10pct"])
                    else "--"
                )
                final_pct = (
                    f"{row['final_bounds_pct']:.2f}"
                    if pd.notna(row["final_bounds_pct"])
                    else "--"
                )
                print(
                    f"{depth} & {num_params} & {runtime} & {iterations} & {time_10} & {final_pct} \\\\"
                )
            print(r"\bottomrule")
            print(r"\end{tabular}")
        else:
            print(
                _format_table(
                    depth_varying_display,
                    [
                        "depth",
                        "num_params",
                        "runtime",
                        "iterations",
                        "time_to_10pct",
                        "final_bounds_pct",
                    ],
                    ["Depth", "#Params", "Runtime (s)", "Iterations", "Time to 10%", "Final %"],
                )
            )
        print()

    # Full table
    print(f"\n{'-' * 80}")
    print(" FULL RESULTS")
    print(f"{'-' * 80}\n")

    combined_display = combined.copy()
    combined_display.loc[combined_display["timeout"], "runtime"] = np.nan

    if latex:
        print("% Full Network Size Results")
        print(r"\begin{tabular}{lrrrrrrrr}")
        print(r"\toprule")
        print(
            r"\textbf{Network} & \textbf{Width} & \textbf{Depth} & \textbf{\#Params} & \textbf{Runtime (s)} & "
            r"\textbf{Iterations} & \textbf{Time to 10\%} & \textbf{Time to 1\%} & \textbf{Final \%} \\"
        )
        print(r"\midrule")
        for _, row in combined_display.iterrows():
            network = row["network"]
            width = int(row["width"]) if pd.notna(row["width"]) else "--"
            depth = int(row["depth"]) if pd.notna(row["depth"]) else "--"
            num_params = (
                f"{int(row['num_params']):,}" if pd.notna(row["num_params"]) else "--"
            )
            runtime = f"{row['runtime']:.2f}" if pd.notna(row["runtime"]) else "--"
            iterations = (
                f"{int(row['iterations'])}" if pd.notna(row["iterations"]) else "--"
            )
            time_10 = (
                f"{row['time_to_10pct']:.2f}"
                if pd.notna(row["time_to_10pct"])
                else "--"
            )
            time_1 = (
                f"{row['time_to_1pct']:.2f}" if pd.notna(row["time_to_1pct"]) else "--"
            )
            final_pct = (
                f"{row['final_bounds_pct']:.2f}"
                if pd.notna(row["final_bounds_pct"])
                else "--"
            )
            print(
                f"{network} & {width} & {depth} & {num_params} & {runtime} & {iterations} & "
                f"{time_10} & {time_1} & {final_pct} \\\\"
            )
        print(r"\bottomrule")
        print(r"\end{tabular}")
    else:
        print(
            _format_table(
                combined_display,
                [
                    "network",
                    "width",
                    "depth",
                    "num_params",
                    "runtime",
                    "iterations",
                    "time_to_10pct",
                    "time_to_1pct",
                    "final_bounds_pct",
                ],
                [
                    "Network",
                    "Width",
                    "Depth",
                    "#Params",
                    "Runtime (s)",
                    "Iterations",
                    "Time to 10%",
                    "Time to 1%",
                    "Final %",
                ],
            )
        )
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse network_size_experiment outputs."
    )
    parser.add_argument(
        "data_dirs",
        type=Path,
        nargs="+",
        help="Directories containing network_size_experiment outputs.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress verbose loading output.",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Output tables as LaTeX code.",
    )
    args = parser.parse_args()
    main(args.data_dirs, quiet=args.quiet, latex=args.latex)
