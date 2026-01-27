# Copyright 2025 David Boetius
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import yaml


def _get_runtime_at(condition: np.ndarray, iter_times: np.ndarray) -> float | None:
    idx = np.where(condition)[0]
    if len(idx) == 0:
        return None
    return iter_times[idx[0]]


def _extract_dataset_name(info_path: Path) -> str:
    parent = info_path.parent
    if parent.name.startswith("repeatition_") or parent.name.startswith("input_"):
        parent = parent.parent
    dataset, *_ = parent.name.split("-")
    return dataset


def _load_bab_run(info_path: Path) -> dict | None:
    print("Loading", info_path)

    with info_path.open("r") as f:
        info = yaml.safe_load(f)
    num_features = len(info["config"]["multi_shap_bab"]["features"])
    num_effective_features = info["config"]["further_stats"].get(
        "num_non_baseline_features", None
    )
    model_output = info["config"]["further_stats"]["model_output"]
    max_iters_reached = info.get("overall", {}).get("max_iters", False)
    timeout_reached = info.get("overall", {}).get("timeout", False)
    overall_rt = info.get("overall", {}).get("runtime", None)

    dataset = _extract_dataset_name(info_path)
    bounds_path = info_path.parent / "multi_shap_bab_bounds.feather"
    if not bounds_path.exists():
        return {
            "dataset": dataset,
            "num_features": num_features,
            "num_effective_features": num_effective_features,
            "max_iters_reached": max_iters_reached,
            "timeout_reached": timeout_reached,
            "overall": overall_rt,
            "iterations": info.get("overall", {}).get("iterations", None),
        }
    bounds = pd.read_feather(bounds_path)

    iter_times = bounds["runtime"]
    lbs = np.array([bounds[(f"{i}", "lb")] for i in range(num_features)]).T
    ubs = np.array([bounds[(f"{i}", "ub")] for i in range(num_features)]).T

    lb_vs_each_ub = np.expand_dims(lbs, -1) > np.expand_dims(ubs, -2)
    some_separated = lb_vs_each_ub.any(axis=(-1, -2))
    some_separated_time = _get_runtime_at(some_separated, iter_times)

    ref_vals1 = np.abs(model_output)
    ranges_norm = ((ubs - lbs) / 2) / ref_vals1.reshape(-1, 1)
    max_norm_ranges = ranges_norm.max(axis=-1)
    max_norm1_ran_lt_10percent = _get_runtime_at(max_norm_ranges <= 0.1, iter_times)
    max_norm1_ran_lt_1percent = _get_runtime_at(max_norm_ranges <= 0.01, iter_times)
    max_norm1_ran_lt_1permille = _get_runtime_at(max_norm_ranges <= 0.001, iter_times)

    ref_vals2 = np.max(np.abs((lbs + ubs) / 2), axis=-1)
    ranges_norm = ((ubs - lbs) / 2) / ref_vals2.reshape(-1, 1)
    max_norm_ranges = ranges_norm.max(axis=-1)
    max_norm2_ran_lt_10percent = _get_runtime_at(max_norm_ranges <= 0.1, iter_times)
    max_norm2_ran_lt_1percent = _get_runtime_at(max_norm_ranges <= 0.01, iter_times)
    max_norm2_ran_lt_1permille = _get_runtime_at(max_norm_ranges <= 0.001, iter_times)

    exact_bounds = None
    if not max_iters_reached and not timeout_reached:
        exact_bounds = iter_times.iloc[-1]

    return {
        "dataset": dataset,
        "num_features": num_features,
        "num_effective_features": num_effective_features,
        "max_iters_reached": max_iters_reached,
        "timeout_reached": timeout_reached,
        "overall": overall_rt,
        "iterations": info.get("overall", {}).get("iterations", None),
        "exact_bounds": exact_bounds,
        "some_separated": some_separated_time,
        "max_norm_to_out_ran_lt_10percent": max_norm1_ran_lt_10percent,
        "max_norm_to_out_ran_lt_1percent": max_norm1_ran_lt_1percent,
        "max_norm_to_out_ran_lt_1permille": max_norm1_ran_lt_1permille,
        "max_norm_to_maxmid_ran_lt_10percent": max_norm2_ran_lt_10percent,
        "max_norm_to_maxmid_ran_lt_1percent": max_norm2_ran_lt_1percent,
        "max_norm_to_maxmid_ran_lt_1permille": max_norm2_ran_lt_1permille,
    }


def _load_exactshap_run(info_path: Path) -> dict | None:
    print("Loading", info_path)

    with info_path.open("r") as f:
        info = yaml.safe_load(f)

    dataset = _extract_dataset_name(info_path)

    runtime = info.get("overall", {}).get("runtime", None)
    iterations = info.get("overall", {}).get("iterations", None)
    return {"dataset": dataset, "overall": runtime, "iterations": iterations}


def _iter_runs(data_dir: Path, load_run) -> Iterable[dict]:
    for info_path in data_dir.rglob("info.yaml"):
        run = load_run(info_path)
        if run is not None:
            yield run


def _load_data_dir(data_dir: Path) -> pd.DataFrame:
    """Load runs from a data directory without aggregation (one row per run)."""
    bab_runs_list = list(_iter_runs(data_dir / "BaB", _load_bab_run))
    exactshap_runs_list = list(_iter_runs(data_dir / "ExactSHAP", _load_exactshap_run))

    if not bab_runs_list and not exactshap_runs_list:
        raise SystemExit(f"No runs found in {data_dir}")

    rows = []

    # Process bab runs - one row per run
    for run in bab_runs_list:
        row = {
            "dataset": run["dataset"],
            "num_features": run.get("num_features"),
            "num_effective_features": run.get("num_effective_features"),
        }
        for key, value in run.items():
            if key not in ["dataset", "num_features", "num_effective_features"]:
                row[f"bab_{key}"] = value
        rows.append(row)

    # Process exactshap runs - merge with existing rows by dataset if possible
    exactshap_by_dataset: dict[str, list[dict]] = defaultdict(list)
    for run in exactshap_runs_list:
        exactshap_by_dataset[run["dataset"]].append(run)

    # Match exactshap runs to bab runs by dataset
    dataset_indices: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        dataset_indices[row["dataset"]].append(i)

    for dataset, exactshap_runs in exactshap_by_dataset.items():
        indices = dataset_indices.get(dataset, [])
        for j, run in enumerate(exactshap_runs):
            if j < len(indices):
                # Merge with existing bab row
                idx = indices[j]
                rows[idx]["exactshap"] = run.get("overall")
                rows[idx]["exactshap_iterations"] = run.get("iterations")
            else:
                # No matching bab row, create new row
                rows.append({
                    "dataset": dataset,
                    "exactshap": run.get("overall"),
                    "exactshap_iterations": run.get("iterations"),
                })

    return pd.DataFrame(rows)


def _compute_runtime_statistics(
    concat: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Compute all runtime statistics (aggregated once).

    Returns:
        Tuple of (stats dict, success_freq DataFrame)
    """
    # Convert everything that looks numeric (including nullable ints/bools) so that
    # quantiles/means/stds ignore textual columns instead of raising.
    numeric_concat = concat.apply(pd.to_numeric, errors="coerce")
    grouped_numeric = numeric_concat.groupby(concat["dataset"])

    # Compute all statistics
    median = grouped_numeric.median()
    q1 = grouped_numeric.quantile(0.25, interpolation="linear")
    q3 = grouped_numeric.quantile(0.75, interpolation="linear")
    iqr = q3 - q1
    mean = grouped_numeric.mean()
    std = grouped_numeric.std()
    min_val = grouped_numeric.min()
    max_val = grouped_numeric.max()

    # Compute success frequencies (proportion of non-null values)
    count_non_null = grouped_numeric.count()
    count_total = grouped_numeric.size()
    success_freq = count_non_null.div(count_total, axis=0)

    # Get num_features and num_effective_features for sorting (use first value)
    grouped_first = concat.groupby("dataset").first()
    sort_cols = grouped_first[["num_features", "num_effective_features"]].apply(
        pd.to_numeric, errors="coerce"
    )

    stats = {
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "mean": mean,
        "std": std,
        "min": min_val,
        "max": max_val,
    }

    # Sort all by num_effective_features, num_features
    for name, df in stats.items():
        df["_sort_eff"] = sort_cols["num_effective_features"]
        df["_sort_feat"] = sort_cols["num_features"]
        stats[name] = df.sort_values(by=["_sort_eff", "_sort_feat"]).drop(
            columns=["_sort_eff", "_sort_feat"]
        )

    # Sort success_freq as well
    success_freq["_sort_eff"] = sort_cols["num_effective_features"]
    success_freq["_sort_feat"] = sort_cols["num_features"]
    success_freq = success_freq.sort_values(by=["_sort_eff", "_sort_feat"]).drop(
        columns=["_sort_eff", "_sort_feat"]
    )

    return stats, success_freq


def _save_runtimes_csv(
    stats: dict[str, pd.DataFrame], success_freq: pd.DataFrame, output_path: Path
) -> None:
    """Save all statistics to a single CSV file with multi-level columns."""
    # Create a combined DataFrame with statistic as top-level column
    combined_parts = []
    for stat_name, df in stats.items():
        df_copy = df.copy()
        df_copy.columns = pd.MultiIndex.from_product(
            [[stat_name], df_copy.columns], names=["statistic", "metric"]
        )
        combined_parts.append(df_copy)

    # Add success frequencies
    freq_copy = success_freq.copy()
    freq_copy.columns = pd.MultiIndex.from_product(
        [["success_freq"], freq_copy.columns], names=["statistic", "metric"]
    )
    combined_parts.append(freq_copy)

    combined = pd.concat(combined_parts, axis=1)
    combined.to_csv(output_path)
    print(f"\nSaved statistics to {output_path}")


def _print_stat_table(
    name: str, df: pd.DataFrame, display_cols: list[str], col_names: list[str]
) -> None:
    """Print a single statistic table."""
    # Filter to available columns
    available_cols = [col for col in display_cols if col in df.columns]
    if not available_cols:
        return

    short_df = df.loc[:, available_cols].copy()
    short_df.columns = col_names[: len(available_cols)]

    print()
    print(name)
    print("-" * 20)
    print(short_df)


def main(data_dirs: Sequence[Path]) -> None:
    # Load raw data from all directories (no aggregation yet)
    data = [_load_data_dir(data_dir) for data_dir in data_dirs]
    concat = pd.concat(data, ignore_index=True)

    # Compute all statistics (aggregation happens once here)
    stats, success_freq = _compute_runtime_statistics(concat)

    # Save to CSV
    output_path = data_dirs[0] / "runtimes.csv"
    _save_runtimes_csv(stats, success_freq, output_path)

    # Display settings
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.precision", 0)
    pd.set_option("display.width", 100)

    # Columns to display for runtime stats
    display_cols = [
        "num_features",
        "num_effective_features",
        "exactshap",
        "bab_max_norm_to_out_ran_lt_10percent",
        "bab_max_norm_to_out_ran_lt_1percent",
        "bab_max_norm_to_out_ran_lt_1permille",
        "bab_exact_bounds",
    ]
    col_names = [
        "#features",
        "#effective",
        "exactshap",
        "bab 10%",
        "bab 1%",
        "bab 0.1%",
        "bab exact",
    ]

    # Print each statistic in a separate table
    stat_display_names = {
        "median": "Median",
        "q1": "Q1",
        "q3": "Q3",
        "iqr": "IQR (Q3 - Q1)",
        "mean": "Mean",
        "std": "Std Dev",
        "min": "Min",
        "max": "Max",
    }

    for stat_key, display_name in stat_display_names.items():
        _print_stat_table(display_name, stats[stat_key], display_cols, col_names)

    # Print success frequencies
    pd.set_option("display.precision", 2)
    _print_stat_table(
        "Success Frequency (proportion of runs achieving milestone)",
        success_freq,
        display_cols,
        col_names,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse compare_to_exactshap outputs."
    )
    parser.add_argument(
        "data_dirs",
        type=Path,
        nargs="+",
        help="Directories containing compare_to_exactshap outputs.",
    )
    main(parser.parse_args().data_dirs)
