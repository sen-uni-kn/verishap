# Copyright 2025 David Boetius
import argparse
from pathlib import Path
from typing import Iterable, List, TypedDict

import pandas as pd
import yaml


class RunRecord(TypedDict):
    compute_bounds: str
    select_strategy: str
    split_strategy: str
    repetition: str
    iterations: int
    runtime: float | None


def _read_iterations(info_path: Path, info: dict) -> int | None:
    try:
        return int(info["multi_shap_bab"]["iterations"])
    except (KeyError, TypeError):
        pass

    iter_stats_path = info_path.parent / "multi_shap_bab_iter_stats.yaml"
    if not iter_stats_path.exists():
        return None

    with iter_stats_path.open("r") as f:
        iter_stats = yaml.safe_load(f)

    if not isinstance(iter_stats, dict) or not iter_stats:
        return None

    try:
        max_iter = max(int(k) for k in iter_stats.keys())
    except (ValueError, TypeError):
        return None

    return max_iter + 1


def _load_run(info_path: Path, repetition: str) -> RunRecord | None:
    with info_path.open("r") as f:
        info = yaml.safe_load(f)

    try:
        cfg = info["config"]["multi_shap_bab"]
    except (KeyError, TypeError):
        return None

    iterations = _read_iterations(info_path, info)
    if iterations is None:
        return None

    runtime = None
    try:
        runtime = float(info.get("overall", {}).get("runtime"))
    except (TypeError, ValueError):
        runtime = None

    return {
        "compute_bounds": cfg["compute_bounds"],
        "select_strategy": cfg["select_strategy"],
        "split_strategy": cfg["split_strategy"],
        "repetition": repetition,
        "iterations": iterations,
        "runtime": runtime,
    }


def _iter_runs(data_dir: Path) -> Iterable[RunRecord]:
    for repetition_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        repetition = repetition_dir.name
        for info_path in repetition_dir.rglob("info.yaml"):
            run = _load_run(info_path, repetition)
            if run is not None:
                yield run


def main(data_dir: Path) -> None:
    runs: List[RunRecord] = list(_iter_runs(data_dir))

    if not runs:
        raise SystemExit(f"No runs found in {data_dir}")

    df = pd.DataFrame(runs)
    df = df.set_index(
        ["split_strategy", "compute_bounds", "select_strategy", "repetition"]
    ).sort_index()

    grouped = df.groupby(
        ["compute_bounds", "select_strategy", "split_strategy"]
    ).agg(
        iterations_median=("iterations", "median"),
        iterations_min=("iterations", "min"),
        iterations_max=("iterations", "max"),
        repetitions=("iterations", "count"),
        runtime_median=("runtime", "median"),
        runtime_min=("runtime", "min"),
        runtime_max=("runtime", "max"),
    ).sort_index()

    runtime_summary = grouped["runtime_median"]

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    print("\nIterations per configuration (min/median/max across repetitions):")
    print(grouped)
    print("\nMedian runtime per configuration:")
    print(runtime_summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse multi_shap_bab iterations across heuristics."
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Directory containing compare_heuristics outputs (patches_*).",
    )
    main(parser.parse_args().data_dir)
