# Copyright 2025 David Boetius
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import yaml

VERBOSE = True
TIMEOUT = 400  # Runtime threshold in seconds - values above this indicate a timeout

# LaTeX name mappings - only names in these dictionaries will be included in LaTeX output
STRATEGY_LATEX_MAPPING = {
    "alpha-crown": r"\AlphaCROWN",
    "crown": r"\CROWN",
    "crown_ibp": r"\CROWNIBP",
    "ibp": r"\IBP",
    "max-diam": r"\SelectMaxDiam",
    "min-diam": r"\SelectMinDiam",
    "longest-edge": r"\SplitInOrder",
    "strong-branching-worse": r"\SplitStrongBranching",
    "smart-branching-ibp-worse": r"\SplitSmartBranchingIBP",
    "smears": r"\SplitSmears",
}

NETWORK_LATEX_MAPPING = {
    "german": r"\GermanDataset",
    "mushroom": r"\MushroomDataset",
    "default": r"\DefaultDataset",
    "automobile": r"\AutomobileDataset",
    "steel": r"\SteelDataset",
    "sonar": r"\SonarDataset",
}


def _strategy_to_latex(strategy: str) -> str:
    """Convert strategy name to LaTeX command."""
    return STRATEGY_LATEX_MAPPING.get(strategy, strategy)


def _network_to_latex(network: str) -> str:
    """Convert network name to LaTeX command.

    Matches network names by checking if any key in NETWORK_LATEX_MAPPING
    appears as a substring in the network name.
    """
    network_lower = str(network).lower()
    for key, latex_cmd in NETWORK_LATEX_MAPPING.items():
        if key in network_lower:
            return latex_cmd
    return network


def _network_has_latex_mapping(network: str) -> bool:
    """Check if a network name has a LaTeX mapping."""
    network_lower = str(network).lower()
    return any(key in network_lower for key in NETWORK_LATEX_MAPPING.keys())


def _format_latex_simple_table(df: pd.DataFrame) -> str:
    """Format a simple DataFrame as LaTeX table rows with proper alignment.

    Used for success rate tables that aren't pivot tables.

    Args:
        df: DataFrame to format
    """
    if df.empty:
        return ""

    df = df.copy()

    # Format all values
    formatted_data = []
    for _, row in df.iterrows():
        row_values = []
        for val in row:
            if pd.isna(val):
                row_values.append("--")
            elif isinstance(val, (int, float)):
                row_values.append(f"{val:.2f}")
            else:
                row_values.append(str(val))
        formatted_data.append(row_values)

    # Calculate max width for each column
    col_widths = []
    for col_idx in range(len(df.columns)):
        header_width = len(str(df.columns[col_idx]))
        data_width = max(
            len(formatted_data[row_idx][col_idx])
            for row_idx in range(len(formatted_data))
        )
        col_widths.append(max(header_width, data_width))

    # Build header row
    header_parts = []
    for col_idx, col_name in enumerate(df.columns):
        header_text = r"\textbf{" + str(col_name) + "}"
        # Left-align first column (names), right-align numeric columns
        if col_idx == 0:
            header_parts.append(header_text.ljust(len(header_text)))
        else:
            header_parts.append(header_text)

    lines = []
    lines.append(" & ".join(header_parts) + r" \\")
    lines.append(r"\midrule")

    # Build data rows
    for row_idx in range(len(formatted_data)):
        row_parts = []
        for col_idx, val_str in enumerate(formatted_data[row_idx]):
            # Left-align first column (names), right-align numeric columns
            if col_idx == 0:
                row_parts.append(val_str.ljust(col_widths[col_idx]))
            else:
                row_parts.append(val_str.rjust(col_widths[col_idx]))
        lines.append(" & ".join(row_parts) + r" \\")

    return "\n".join(lines)


def _format_latex_table(
    df: pd.DataFrame, bold_min: bool = False, format_str: str = ":.2f"
) -> str:
    """Format a DataFrame as LaTeX table with custom formatting.

    Args:
        df: DataFrame to format
        bold_min: If True, bold the minimum value in each row
        format_str: Format string for numeric values (e.g., ":.2f" or ":.0f")
    """
    if df.empty:
        return ""

    df = df.copy()

    # Filter columns to only include those in STRATEGY_LATEX_MAPPING
    valid_cols = [col for col in df.columns if str(col) in STRATEGY_LATEX_MAPPING]
    if not valid_cols:
        return ""
    df = df[valid_cols]

    # Filter rows to only include those with NETWORK_LATEX_MAPPING
    valid_rows = [idx for idx in df.index if _network_has_latex_mapping(idx)]
    if not valid_rows:
        return ""
    df = df.loc[valid_rows]

    # Replace strategy names in column headers and network names
    df.columns = [_strategy_to_latex(str(col)) for col in df.columns]
    network_names = [_network_to_latex(str(idx)) for idx in df.index]

    # Format all data values and collect for width calculation
    formatted_data = []
    for idx, row in df.iterrows():
        row_values = []
        for val in row:
            if pd.isna(val):
                row_values.append("--")
            else:
                row_values.append(format(val, format_str[1:]))
        formatted_data.append(row_values)

    # Apply bold formatting to minimum values if requested
    if bold_min:
        for row_idx, row in enumerate(df.itertuples(index=False)):
            numeric_vals = [v for v in row if not pd.isna(v)]
            if numeric_vals:
                min_val = min(numeric_vals)
                for col_idx, val in enumerate(row):
                    if not pd.isna(val) and val == min_val:
                        formatted_data[row_idx][col_idx] = (
                            r"\textbf{" + formatted_data[row_idx][col_idx] + "}"
                        )

    # Find max length for network column (using actual string length)
    max_network_len = max(len(name) for name in network_names)

    # Find max length for each data column (using actual string length, excluding headers)
    col_widths = []
    for col_idx in range(len(df.columns)):
        data_width = max(
            len(formatted_data[row_idx][col_idx])
            for row_idx in range(len(formatted_data))
        )
        col_widths.append(data_width)

    # Build header row
    header_cols = [r"\textbf{" + str(col) + "}" for col in df.columns]
    header_parts = [r"\textbf{Network}"]
    header_parts.extend(header_cols)

    lines = []
    lines.append(" & ".join(header_parts) + r" \\")
    lines.append(r"\midrule")

    # Build data rows
    for row_idx, network_name in enumerate(network_names):
        # Left-align network name by padding on the right
        row_parts = [network_name.ljust(max_network_len)]

        # Right-align data values by padding on the left
        for col_idx, val_str in enumerate(formatted_data[row_idx]):
            row_parts.append(val_str.rjust(col_widths[col_idx]))

        lines.append(" & ".join(row_parts) + r" \\")

    return "\n".join(lines)


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

    config = info.get("config", {})
    multi_shap_bab_config = config.get("multi_shap_bab", {})
    features = multi_shap_bab_config.get("features", [])
    num_features = len(features)

    # Extract input dimension as length of features list
    input_dim = num_features

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
            ranges_norm = ((ubs - lbs) / 2) / ref_vals.reshape(-1, 1)
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
        "num_features": num_features,
        "input_dim": input_dim,
        "success": success,
        "split_strategy": multi_shap_bab_config.get("split_strategy"),
        "select_strategy": multi_shap_bab_config.get("select_strategy"),
        "compute_bounds": multi_shap_bab_config.get("compute_bounds"),
        "time_to_10pct": time_to_10pct,
        "time_to_1pct": time_to_1pct,
        "final_bounds_pct": final_bounds_pct,
    }


def _parse_path(info_path: Path) -> dict | None:
    """Extract category, strategy, network, and repetition from path."""
    # Expected structure: .../category/strategy/network/repeatition_i/info.yaml
    parts = info_path.parts
    if len(parts) < 5:
        return None

    repetition_dir = parts[-2]
    network = parts[-3]
    strategy = parts[-4]
    category = parts[-5]

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

    return {
        "category": category,
        "strategy": strategy,
        "network": network,
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
            yield run


def _load_data_dir(data_dir: Path) -> dict[str, dict[str, pd.DataFrame]]:
    """Load all runs from a data directory and compute median runtimes."""
    runs = list(_iter_runs(data_dir))

    if not runs:
        raise SystemExit(f"No runs found in {data_dir}")

    df = pd.DataFrame(runs)

    # Compute success rate per strategy
    success_by_strategy = df.groupby(["category", "strategy"]).agg(
        total_runs=("success", "count"),
        successful_runs=("success", "sum"),
    )
    success_by_strategy["success_rate"] = (
        success_by_strategy["successful_runs"] / success_by_strategy["total_runs"]
    )

    # Compute success rate per network
    success_by_network = df.groupby(["category", "network"]).agg(
        total_runs=("success", "count"),
        successful_runs=("success", "sum"),
    )
    success_by_network["success_rate"] = (
        success_by_network["successful_runs"] / success_by_network["total_runs"]
    )

    # Group by category, strategy, network and compute median runtime
    grouped = df.groupby(["category", "strategy", "network"]).agg(
        {
            "runtime": "median",
            "timeout": "any",
            "max_iters": "any",
            "iterations": "median",
            "total_branches": "median",
            "num_features": "first",
            "input_dim": "first",
            "time_to_10pct": "median",
            "time_to_1pct": "median",
            "final_bounds_pct": "median",
        }
    )

    # Create separate DataFrames for each category
    result = {}
    for category in df["category"].unique():
        cat_df = grouped.loc[category].reset_index()

        # Sort by input dimension (put NaN at the end)
        cat_df = cat_df.sort_values("input_dim", na_position="last")

        # Create a mapping from network to input_dim for sorting later
        network_input_dim = cat_df.set_index("network")["input_dim"].to_dict()

        # Create runtime table
        runtime_df = cat_df.pivot(index="network", columns="strategy", values="runtime")

        # Create 10% tightness table
        time_10pct_df = cat_df.pivot(
            index="network", columns="strategy", values="time_to_10pct"
        )

        # Create 1% tightness table
        time_1pct_df = cat_df.pivot(
            index="network", columns="strategy", values="time_to_1pct"
        )

        # Create final bounds percentage table
        final_bounds_df = cat_df.pivot(
            index="network", columns="strategy", values="final_bounds_pct"
        )

        # Create iterations table
        iterations_df = cat_df.pivot(
            index="network", columns="strategy", values="iterations"
        )

        # Get success rate for this category
        if category in success_by_strategy.index.get_level_values(0):
            cat_success_strategy = success_by_strategy.loc[category].reset_index()
        else:
            cat_success_strategy = pd.DataFrame()

        if category in success_by_network.index.get_level_values(0):
            cat_success_network = success_by_network.loc[category].reset_index()
            # Add input_dim for sorting
            cat_success_network["input_dim"] = cat_success_network["network"].map(
                network_input_dim
            )
            cat_success_network = cat_success_network.sort_values(
                "input_dim", na_position="last"
            )
            cat_success_network = cat_success_network.drop(columns=["input_dim"])
        else:
            cat_success_network = pd.DataFrame()

        result[category] = {
            "runtime": runtime_df,
            "time_to_10pct": time_10pct_df,
            "time_to_1pct": time_1pct_df,
            "final_bounds_pct": final_bounds_df,
            "iterations": iterations_df,
            "success_rate_strategy": cat_success_strategy,
            "success_rate_network": cat_success_network,
            "network_input_dim": network_input_dim,
        }

    # Also save the full grouped data
    grouped.to_csv(data_dir / "heuristics_runtimes.csv")

    return result


def main(data_dirs: Sequence[Path], quiet: bool = False, latex: bool = False) -> None:
    global VERBOSE
    VERBOSE = not quiet

    # Load data from all directories
    all_data = defaultdict(lambda: defaultdict(list))
    for data_dir in data_dirs:
        data = _load_data_dir(data_dir)
        for category, dfs_dict in data.items():
            for metric, df in dfs_dict.items():
                all_data[category][metric].append(df)

    # Compute median across all data directories for each category and metric
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.precision", 2)
    pd.set_option("display.width", 200)

    # Collect network to input_dim mapping across all categories
    global_network_input_dim = {}
    for category in all_data.keys():
        if "network_input_dim" in all_data[category]:
            for mapping in all_data[category]["network_input_dim"]:
                global_network_input_dim.update(mapping)

    # Display global success rate by network (aggregated across all categories)
    if latex:
        print(
            "\n% ================================================================================"
        )
        print("% SUCCESS RATE BY NETWORK (ALL CATEGORIES)")
        print(
            "% ================================================================================\n"
        )
    else:
        print(f"\n{'=' * 80}")
        print(f" SUCCESS RATE BY NETWORK (ALL CATEGORIES)")
        print(f"{'=' * 80}\n")

    all_network_success_dfs = []
    for category in all_data.keys():
        if "success_rate_network" in all_data[category]:
            all_network_success_dfs.extend(all_data[category]["success_rate_network"])

    if all_network_success_dfs:
        concat = pd.concat(all_network_success_dfs)
        global_network_success = concat.groupby("network").agg(
            {
                "total_runs": "sum",
                "successful_runs": "sum",
            }
        )
        global_network_success["success_rate"] = (
            global_network_success["successful_runs"]
            / global_network_success["total_runs"]
        )
        global_network_success = global_network_success.reset_index()

        # Sort by input dimension
        if global_network_input_dim:
            global_network_success["_input_dim"] = global_network_success[
                "network"
            ].map(global_network_input_dim)
            global_network_success = global_network_success.sort_values(
                "_input_dim", na_position="last"
            )
            global_network_success = global_network_success.drop(columns=["_input_dim"])

        if latex:
            # Filter to only include networks with LaTeX mappings
            global_network_success_latex = global_network_success.copy()
            global_network_success_latex = global_network_success_latex[
                global_network_success_latex["network"].apply(
                    _network_has_latex_mapping
                )
            ]
            # Replace network names with LaTeX commands
            global_network_success_latex["network"] = global_network_success_latex[
                "network"
            ].apply(_network_to_latex)

            print("% Global Network Success Rate")
            print(_format_latex_simple_table(global_network_success_latex))
        else:
            print(global_network_success.to_string(index=False))
        print()

    for category in sorted(all_data.keys()):
        if latex:
            print(
                f"\n% ================================================================================"
            )
            print(f"% {category.upper().replace('_', ' ')}")
            print(
                f"% ================================================================================"
            )
        else:
            print(f"\n{'=' * 80}")
            print(f" {category.upper().replace('_', ' ')}")
            print(f"{'=' * 80}")

        # Display success rate summary by strategy
        if "success_rate_strategy" in all_data[category]:
            success_dfs = all_data[category]["success_rate_strategy"]
            if success_dfs and any(not df.empty for df in success_dfs):
                if latex:
                    print("\n% Success Rate by Strategy")
                else:
                    print(f"\nSuccess Rate by Strategy:")
                if len(success_dfs) == 1:
                    success_df = success_dfs[0]
                else:
                    concat = pd.concat(success_dfs)
                    success_df = concat.groupby("strategy").agg(
                        {
                            "total_runs": "sum",
                            "successful_runs": "sum",
                        }
                    )
                    success_df["success_rate"] = (
                        success_df["successful_runs"] / success_df["total_runs"]
                    )
                    success_df = success_df.reset_index()

                if not success_df.empty:
                    if latex:
                        # Filter to only include strategies in STRATEGY_LATEX_MAPPING
                        success_df_latex = success_df.copy()
                        success_df_latex = success_df_latex[
                            success_df_latex["strategy"].isin(
                                STRATEGY_LATEX_MAPPING.keys()
                            )
                        ]
                        # Replace strategy names with LaTeX commands
                        if not success_df_latex.empty:
                            success_df_latex["strategy"] = success_df_latex[
                                "strategy"
                            ].apply(_strategy_to_latex)
                            print(_format_latex_simple_table(success_df_latex))
                    else:
                        print(success_df.to_string(index=False))
                    print()

        # Get network to input_dim mapping (combine from all data dirs)
        network_input_dim = {}
        if "network_input_dim" in all_data[category]:
            for mapping in all_data[category]["network_input_dim"]:
                network_input_dim.update(mapping)

        for metric in [
            "runtime",
            "time_to_10pct",
            "time_to_1pct",
            "final_bounds_pct",
            "iterations",
        ]:
            if metric not in all_data[category]:
                continue

            dfs = all_data[category][metric]
            if len(dfs) == 1:
                median_df = dfs[0]
            else:
                concat = pd.concat(dfs)
                median_df = concat.groupby(concat.index).median()

            # Sort by input dimension
            if network_input_dim:
                # Add input_dim as a column for sorting
                median_df["_input_dim"] = median_df.index.map(network_input_dim)
                median_df = median_df.sort_values("_input_dim", na_position="last")
                median_df = median_df.drop(columns=["_input_dim"])

            # Replace runtime values > TIMEOUT with NaN (displayed as --)
            if metric == "runtime":
                median_df = median_df.map(
                    lambda x: np.nan if pd.notna(x) and x > TIMEOUT else x
                )

            # Print metric-specific header
            if latex:
                if metric == "runtime":
                    print("\n% Overall Runtime (median across repetitions, in seconds)")
                elif metric == "time_to_10pct":
                    print(
                        "\n% Time to 10% Tight Bounds (median across repetitions, in seconds)"
                    )
                elif metric == "time_to_1pct":
                    print(
                        "\n% Time to 1% Tight Bounds (median across repetitions, in seconds)"
                    )
                elif metric == "final_bounds_pct":
                    print(
                        "\n% Final Bound Half-Width (median across repetitions, as % of model output)"
                    )
                elif metric == "iterations":
                    print("\n% Iterations (median across repetitions)")
            else:
                if metric == "runtime":
                    print(f"\nOverall Runtime (median across repetitions, in seconds):")
                elif metric == "time_to_10pct":
                    print(
                        f"\nTime to 10% Tight Bounds (median across repetitions, in seconds):"
                    )
                elif metric == "time_to_1pct":
                    print(
                        f"\nTime to 1% Tight Bounds (median across repetitions, in seconds):"
                    )
                elif metric == "final_bounds_pct":
                    print(
                        f"\nFinal Bound Half-Width (median across repetitions, as % of model output):"
                    )
                elif metric == "iterations":
                    print(f"\nIterations (median across repetitions):")

            # Bold minimum for runtime and final_bounds_pct
            bold_min = metric in ["runtime", "final_bounds_pct"]

            # Use integer formatting for iterations and runtime
            format_str = ":.0f" if metric in ["iterations", "runtime"] else ":.2f"

            if latex:
                print(
                    _format_latex_table(
                        median_df, bold_min=bold_min, format_str=format_str
                    )
                )
            else:
                print(median_df.to_string())
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse compare_heuristics outputs.")
    parser.add_argument(
        "data_dirs",
        type=Path,
        nargs="+",
        help="Directories containing compare_heuristics outputs.",
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
        help="Output tables as LaTeX code with custom formatting.",
    )
    args = parser.parse_args()
    main(args.data_dirs, quiet=args.quiet, latex=args.latex)
