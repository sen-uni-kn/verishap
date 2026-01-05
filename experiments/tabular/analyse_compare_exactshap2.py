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


def _load_bab_runtimes(info_path: Path) -> dict:
    bounds_path = info_path.parent / "multi_shap_bab_bounds.feather"
    if not bounds_path.exists():
        return {}
    bounds = pd.read_feather(bounds_path)

    with info_path.open("r") as f:
        info = yaml.safe_load(f)
    num_features = len(info["config"]["multi_shap_bab"]["features"])
    max_iters = info["config"]["cmd_args"].get("max_iters", None)

    iter_times = bounds["runtime"]
    lbs = np.array([bounds[(f"{i}", "lb")] for i in range(num_features)]).T
    ubs = np.array([bounds[(f"{i}", "ub")] for i in range(num_features)]).T

    lb_vs_each_ub = np.expand_dims(lbs, -1) > np.expand_dims(ubs, -2)
    some_separated = lb_vs_each_ub.any(axis=(-1, -2))
    some_separated_time = _get_runtime_at(some_separated, iter_times)

    ref_vals = np.max(np.abs((lbs + ubs) / 2), axis=-1)
    ranges_norm = ((ubs - lbs) / 2) / ref_vals.reshape(-1, 1)
    max_norm_ranges = ranges_norm.max(axis=-1)
    max_norm_ran_lt_10percent = _get_runtime_at(max_norm_ranges <= 0.1, iter_times)
    max_norm_ran_lt_1percent = _get_runtime_at(max_norm_ranges <= 0.01, iter_times)
    max_norm_ran_lt_1permille = _get_runtime_at(max_norm_ranges <= 0.001, iter_times)

    exact_bounds = None
    if max_iters is not None:
        if int(max_iters) - 1 > len(iter_times):
            exact_bounds = iter_times.iloc[-1]

    return {
        "exact_bounds": exact_bounds,
        "some_separated": some_separated_time,
        "max_norm_ran_lt_10percent": max_norm_ran_lt_10percent,
        "max_norm_ran_lt_1percent": max_norm_ran_lt_1percent,
        "max_norm_ran_lt_1permille": max_norm_ran_lt_1permille,
    }


def _load_bab_run(info_path: Path) -> dict | None:
    with info_path.open("r") as f:
        info = yaml.safe_load(f)

    print("Loading", info_path)
    effective_features = int(info_path.parent.name)

    runtimes = _load_bab_runtimes(info_path)
    return {"effective_features": effective_features, **runtimes}


def _load_exactshap_run(info_path: Path) -> dict | None:
    with info_path.open("r") as f:
        info = yaml.safe_load(f)

    print("Loading", info_path)
    effective_features = int(info_path.parent.name)

    runtime = info.get("overall", {}).get("runtime")
    return {"effective_features": effective_features, "overall": runtime}


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
        dim = run["effective_features"]
        data[dim]["input_dim"] = dim
        for key, value in run.items():
            if key != "effective_features":
                data[dim][f"bab_{key}"] = value
    for run in exactshap_runs:
        dim = run["effective_features"]
        data[dim]["exactshap_overall"] = run["overall"]

    df = pd.DataFrame.from_dict(data, orient="index")
    df = df.sort_values(by="input_dim")

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    print(df)

    df.to_csv(data_dir / "runtimes.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse compare_to_exactshap2 outputs."
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Directory containing compare_heuristics outputs.",
    )
    main(parser.parse_args().data_dir)
