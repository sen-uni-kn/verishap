# Copyright 2025 David Boetius
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml


def _get_runtime_at(condition: np.ndarray, iter_times: np.ndarray) -> float | None:
    idx = np.where(condition)[0]
    if len(idx) == 0:
        return None
    return iter_times[idx[0]]


def _load_bab_run(info_path: Path) -> dict | None:
    print("Loading", info_path)

    with info_path.open("r") as f:
        info = yaml.safe_load(f)
    num_features = len(info["config"]["multi_shap_bab"]["features"])
    num_effective_features = info["config"]["further_stats"][
        "num_non_baseline_features"
    ]
    model_output = info["config"]["further_stats"]["model_output"]
    max_iters_reached = info.get("overall", {}).get("max_iters", False)
    timeout_reached = info.get("overall", {}).get("timeout", False)
    overall_rt = info.get("overall", {}).get("runtime", None)

    num_patches = int(info_path.parent.name.split("_")[0])
    bounds_path = info_path.parent / "multi_shap_bab_bounds.feather"
    if not bounds_path.exists():
        return {
            "dataset": dataset,
            "num_features": num_features,
            "num_effective_features": num_effective_features,
            "max_iters_reached": max_iters_reached,
            "timeout_reached": timeout_reached,
            "overall": overall_rt,
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
        "num_patches": f"{num_patches}x{num_patches}",
        "num_features": num_features,
        "num_effective_features": num_effective_features,
        "max_iters_reached": max_iters_reached,
        "timeout_reached": timeout_reached,
        "overall": overall_rt,
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

    dataset, *_ = info_path.parent.name.split("-")

    runtime = info.get("overall", {}).get("runtime", None)
    return {"dataset": dataset, "overall": runtime}


def _iter_runs(data_dir: Path, load_run) -> Iterable[dict]:
    for info_path in data_dir.rglob("info.yaml"):
        run = load_run(info_path)
        if run is not None:
            yield run


def main(data_dir: Path) -> None:
    bab_runs = list(_iter_runs(data_dir / "BaB", _load_bab_run))
    exactshap_runs = list(_iter_runs(data_dir / "ExactSHAP", _load_exactshap_run))

    if not bab_runs and not exactshap_runs:
        raise SystemExit(f"No runs found in {data_dir}")

    data = defaultdict(dict)
    for run in bab_runs:
        dataset = run["dataset"]
        data[dataset]["num_features"] = run["num_features"]
        data[dataset]["num_effective_features"] = run["num_effective_features"]
        for key, value in run.items():
            if key not in ["dataset", "num_features", "num_effective_features"]:
                data[dataset][f"bab_{key}"] = value
    for run in exactshap_runs:
        dataset = run["dataset"]
        data[dataset]["exactshap"] = run["overall"]

    df = pd.DataFrame.from_dict(data, orient="index")
    df = df.sort_values(by="num_effective_features")

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    print(df)

    df.to_csv(data_dir / "runtimes.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse compare_to_exactshap outputs."
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Directory containing compare_heuristics outputs.",
    )
    main(parser.parse_args().data_dir)
