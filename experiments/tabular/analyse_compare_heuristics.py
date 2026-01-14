# Copyright 2025 David Boetius
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import yaml

VERBOSE = True


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

    runtime = info.get("overall", {}).get("runtime", None)
    timeout = info.get("overall", {}).get("timeout", False)
    max_iters = info.get("overall", {}).get("max_iters", False)
    iterations = info.get("multi_shap_bab", {}).get("iterations", None)
    total_branches = info.get("multi_shap_bab", {}).get("total_branches", None)

    config = info.get("config", {})
    multi_shap_bab_config = config.get("multi_shap_bab", {})
    num_features = len(multi_shap_bab_config.get("features", []))

    # Initialize bound timing metrics
    time_to_10pct = None
    time_to_1pct = None

    # Load bounds if available
    bounds_path = info_path.parent / "multi_shap_bab_bounds.feather"
    if bounds_path.exists():
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

    return {
        "runtime": runtime,
        "timeout": timeout,
        "max_iters": max_iters,
        "iterations": iterations,
        "total_branches": total_branches,
        "num_features": num_features,
        "split_strategy": multi_shap_bab_config.get("split_strategy"),
        "select_strategy": multi_shap_bab_config.get("select_strategy"),
        "compute_bounds": multi_shap_bab_config.get("compute_bounds"),
        "time_to_10pct": time_to_10pct,
        "time_to_1pct": time_to_1pct,
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

    # Group by category, strategy, network and compute median runtime
    grouped = df.groupby(["category", "strategy", "network"]).agg(
        {
            "runtime": "median",
            "timeout": "any",
            "max_iters": "any",
            "iterations": "median",
            "total_branches": "median",
            "num_features": "first",
            "time_to_10pct": "median",
            "time_to_1pct": "median",
        }
    )

    # Create separate DataFrames for each category
    result = {}
    for category in df["category"].unique():
        cat_df = grouped.loc[category].reset_index()

        # Create runtime table
        runtime_df = cat_df.pivot(index="network", columns="strategy", values="runtime")

        # Create 10% tightness table
        time_10pct_df = cat_df.pivot(index="network", columns="strategy", values="time_to_10pct")

        # Create 1% tightness table
        time_1pct_df = cat_df.pivot(index="network", columns="strategy", values="time_to_1pct")

        result[category] = {
            "runtime": runtime_df,
            "time_to_10pct": time_10pct_df,
            "time_to_1pct": time_1pct_df,
        }

    # Also save the full grouped data
    grouped.to_csv(data_dir / "heuristics_runtimes.csv")

    return result


def main(data_dirs: Sequence[Path], quiet: bool = False) -> None:
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

    for category in sorted(all_data.keys()):
        print(f"\n{'='*80}")
        print(f" {category.upper().replace('_', ' ')}")
        print(f"{'='*80}")

        for metric in ["runtime", "time_to_10pct", "time_to_1pct"]:
            if metric not in all_data[category]:
                continue

            dfs = all_data[category][metric]
            if len(dfs) == 1:
                median_df = dfs[0]
            else:
                concat = pd.concat(dfs)
                median_df = concat.groupby(concat.index).median()

            # Print metric-specific header
            if metric == "runtime":
                print(f"\nOverall Runtime (median across repetitions):")
            elif metric == "time_to_10pct":
                print(f"\nTime to 10% Tight Bounds (median across repetitions):")
            elif metric == "time_to_1pct":
                print(f"\nTime to 1% Tight Bounds (median across repetitions):")

            print(median_df.to_string())
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse compare_heuristics outputs."
    )
    parser.add_argument(
        "data_dirs",
        type=Path,
        nargs="+",
        help="Directories containing compare_heuristics outputs.",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress verbose loading output.",
    )
    args = parser.parse_args()
    main(args.data_dirs, quiet=args.quiet)
