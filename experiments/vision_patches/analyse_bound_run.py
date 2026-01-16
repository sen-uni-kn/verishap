import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _to_python_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _serialise_stats(stats: dict) -> dict:
    return {key: _to_python_scalar(value) for key, value in stats.items()}


def _get_runtime_at(condition: np.ndarray, iter_times: np.ndarray) -> float | None:
    idx = np.where(condition)[0]
    if len(idx) == 0:
        return None
    return iter_times[idx[0]]


def _get_iteration_at(condition: np.ndarray) -> float | None:
    idx = np.where(condition)[0]
    if len(idx) == 0:
        return None
    return idx[0]


def _load_run_stats(run_dir: Path) -> dict:
    info_path = run_dir / "info.yaml"
    if not info_path.exists():
        raise FileNotFoundError(f"Could not find info.yaml in {run_dir}")

    print("Loading", info_path)
    with info_path.open("r") as f:
        info = yaml.safe_load(f)

    cmd_args = info["config"].get("cmd_args", {})
    num_patches = cmd_args.get("num_patches")
    if num_patches is None:
        num_patches = run_dir.name
    else:
        num_patches = int(num_patches)

    features = info["config"]["multi_shap_bab"]["features"]
    num_features = len(features)
    num_effective_features = info["config"]["further_stats"].get(
        "num_non_baseline_features", None
    )
    model_output = info["config"]["further_stats"]["model_output"]
    overall = info.get("overall", {})
    max_iters_reached = overall.get("max_iters", False)
    timeout_reached = overall.get("timeout", False)
    overall_rt = overall.get("runtime", None)

    bounds_path = run_dir / "multi_shap_bab_bounds.feather"
    if not bounds_path.exists():
        return {
            "num_patches": f"{num_patches}x{num_patches}",
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

    # Check if any pair of features has disjoint bounds.
    lb_vs_each_ub = np.expand_dims(lbs, -1) > np.expand_dims(ubs, -2)
    some_separated = lb_vs_each_ub.any(axis=(-1, -2))
    some_separated_time = _get_runtime_at(some_separated, iter_times)
    some_separated_iter = _get_iteration_at(some_separated)

    ref_vals1 = np.atleast_1d(np.abs(model_output))
    ranges_norm = ((ubs - lbs) / 2) / ref_vals1.reshape(-1, 1)
    max_norm_ranges = ranges_norm.max(axis=-1)
    max_norm1_ran_lt_10percent = _get_runtime_at(max_norm_ranges <= 0.1, iter_times)
    max_norm1_ran_lt_10percent_iter = _get_iteration_at(max_norm_ranges <= 0.1)
    max_norm1_ran_lt_1percent = _get_runtime_at(max_norm_ranges <= 0.01, iter_times)
    max_norm1_ran_lt_1percent_iter = _get_iteration_at(max_norm_ranges <= 0.01)
    max_norm1_ran_lt_1permille = _get_runtime_at(
        max_norm_ranges <= 0.001, iter_times
    )
    max_norm1_ran_lt_1permille_iter = _get_iteration_at(
        max_norm_ranges <= 0.001
    )

    ref_vals2 = np.max(np.abs((lbs + ubs) / 2), axis=-1)
    ranges_norm = ((ubs - lbs) / 2) / ref_vals2.reshape(-1, 1)
    max_norm_ranges = ranges_norm.max(axis=-1)
    max_norm2_ran_lt_10percent = _get_runtime_at(max_norm_ranges <= 0.1, iter_times)
    max_norm2_ran_lt_10percent_iter = _get_iteration_at(max_norm_ranges <= 0.1)
    max_norm2_ran_lt_1percent = _get_runtime_at(max_norm_ranges <= 0.01, iter_times)
    max_norm2_ran_lt_1percent_iter = _get_iteration_at(max_norm_ranges <= 0.01)
    max_norm2_ran_lt_1permille = _get_runtime_at(
        max_norm_ranges <= 0.001, iter_times
    )
    max_norm2_ran_lt_1permille_iter = _get_iteration_at(
        max_norm_ranges <= 0.001
    )

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
        "some_separated_iter": some_separated_iter,
        "max_norm_to_out_ran_lt_10percent": max_norm1_ran_lt_10percent,
        "max_norm_to_out_ran_lt_10percent_iter": max_norm1_ran_lt_10percent_iter,
        "max_norm_to_out_ran_lt_1percent": max_norm1_ran_lt_1percent,
        "max_norm_to_out_ran_lt_1percent_iter": max_norm1_ran_lt_1percent_iter,
        "max_norm_to_out_ran_lt_1permille": max_norm1_ran_lt_1permille,
        "max_norm_to_out_ran_lt_1permille_iter": max_norm1_ran_lt_1permille_iter,
        "max_norm_to_maxmid_ran_lt_10percent": max_norm2_ran_lt_10percent,
        "max_norm_to_maxmid_ran_lt_10percent_iter": max_norm2_ran_lt_10percent_iter,
        "max_norm_to_maxmid_ran_lt_1percent": max_norm2_ran_lt_1percent,
        "max_norm_to_maxmid_ran_lt_1percent_iter": max_norm2_ran_lt_1percent_iter,
        "max_norm_to_maxmid_ran_lt_1permille": max_norm2_ran_lt_1permille,
        "max_norm_to_maxmid_ran_lt_1permille_iter": max_norm2_ran_lt_1permille_iter,
    }


def main(run_dir: Path) -> None:
    stats = _load_run_stats(run_dir)
    df = pd.Series(stats, name="value")
    print(df.to_frame())
    runtime_path = run_dir / "runtime.yaml"
    with runtime_path.open("w") as f:
        yaml.safe_dump(_serialise_stats(stats), f, sort_keys=False)
    print(f"Wrote {runtime_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse a single bound.py run.")
    parser.add_argument(
        "run_dir", type=Path, help="Directory containing the outputs of a bound.py run"
    )
    main(parser.parse_args().run_dir)
