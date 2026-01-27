# Copyright 2025 David Boetius
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

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
    if parent.name.startswith("repeatition_"):
        parent = parent.parent
    dataset, *_ = parent.name.split("-")
    return dataset


def _aggregate_value(values: Sequence[Any]) -> Any:
    present_values = [value for value in values if value is not None]
    if not present_values:
        return None
    first = present_values[0]

    if isinstance(first, (bool, np.bool_)):
        return any(present_values)

    if isinstance(first, (int, np.integer)) and not isinstance(first, bool):
        return int(np.median(present_values))

    if isinstance(first, (float, np.floating, int, np.integer)):
        return float(np.median(present_values))

    return first


def _aggregate_runs_by_dataset(runs: Iterable[dict]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        dataset = run["dataset"]
        for key, value in run.items():
            if key == "dataset":
                continue
            grouped[dataset][key].append(value)
    aggregated = {
        dataset: {key: _aggregate_value(values) for key, values in metrics.items()}
        for dataset, metrics in grouped.items()
    }
    return aggregated


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


def _load_data_dir(data_dir: Path) -> dict:
    bab_runs_list = list(_iter_runs(data_dir / "BaB", _load_bab_run))
    exactshap_runs_list = list(_iter_runs(data_dir / "ExactSHAP", _load_exactshap_run))

    if not bab_runs_list and not exactshap_runs_list:
        raise SystemExit(f"No runs found in {data_dir}")

    data = defaultdict(dict)

    bab_runs = _aggregate_runs_by_dataset(bab_runs_list)
    for dataset, metrics in bab_runs.items():
        data[dataset]["num_features"] = metrics.get("num_features")
        data[dataset]["num_effective_features"] = metrics.get("num_effective_features")
        for key, value in metrics.items():
            if key not in ["num_features", "num_effective_features"]:
                data[dataset][f"bab_{key}"] = value

    exactshap_runs = _aggregate_runs_by_dataset(exactshap_runs_list)
    for dataset, metrics in exactshap_runs.items():
        runtime = metrics.get("overall")
        if runtime is not None:
            data[dataset]["exactshap"] = runtime
        iterations = metrics.get("iterations")
        if iterations is not None:
            data[dataset]["exactshap_iterations"] = iterations

    df = pd.DataFrame.from_dict(data, orient="index")
    df = df.sort_values(by=["num_effective_features", "num_features"])
    df.to_csv(data_dir / "runtimes.csv")
    return df


def main(data_dirs: Sequence[Path]) -> None:
    data = [_load_data_dir(data_dir) for data_dir in data_dirs]
    concat = pd.concat(data)
    median = concat.groupby(concat.index).median()
    median = median.sort_values(by=["num_effective_features", "num_features"])

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.precision", 0)
    pd.set_option("display.width", 100)
    short_df = median.loc[
        :,
        [
            "num_features",
            "num_effective_features",
            "exactshap",
            "bab_max_norm_to_out_ran_lt_10percent",
            "bab_max_norm_to_out_ran_lt_1percent",
            "bab_max_norm_to_out_ran_lt_1permille",
            "bab_exact_bounds",
        ],
    ]
    short_df.columns = [
        "#features",
        "#effective",
        "exactshap",
        "bab 10%",
        "bab 1%",
        "bab 0.1%",
        "bab exact",
    ]
    print(short_df)


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
