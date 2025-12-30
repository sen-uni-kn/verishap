# Copyright 2025 David Boetius
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


def _load_bab_run(info_path: Path) -> dict | None:
    with info_path.open("r") as f:
        info = yaml.safe_load(f)

    print("Loading", info_path)
    dataset, *_ = info_path.parent.name.split("-")
    dataset_dim = len(info["config"]["multi_shap_bab"]["features"])

    timeout = info.get("overall", {}).get("runtime", {}).get("timeout", True)
    overall_rt = info.get("overall", {}).get("runtime", {}).get("overall")
    some_separated_rt = info.get("overall", {}).get("runtime", {}).get("some_separated")
    all_separated_rt = info.get("overall", {}).get("runtime", {}).get("all_separated")
    largest_shap_rt = info.get("overall", {}).get("runtime", {}).get("largest_shap")
    smallest_shap_rt = info.get("overall", {}).get("runtime", {}).get("smallest_shap")
    max_range_less_than_1_rt = info.get("overall", {}).get("runtime", {}).get("max_range_less_than_1")
    max_range_less_than_0_01_rt = info.get("overall", {}).get("runtime", {}).get("max_range_less_than_0.01")
    max_range_less_than_0_001_rt = info.get("overall", {}).get("runtime", {}).get("max_range_less_than_0.001")
    max_range_less_than_0_0001_rt = info.get("overall", {}).get("runtime", {}).get("max_range_less_than_0.0001")
    max_range_less_than_0_00001_rt = info.get("overall", {}).get("runtime", {}).get("max_range_less_than_0.00001")
    max_range_less_than_0_000001_rt = info.get("overall", {}).get("runtime", {}).get("max_range_less_than_0.000001")
    max_range_less_than_0_0000001_rt = info.get("overall", {}).get("runtime", {}).get("max_range_less_than_0.0000001")
    max_range_less_than_0_00000001_rt = info.get("overall", {}).get("runtime", {}).get("max_range_less_than_0.00000001")

    if timeout:
        overall_rt = 900
        if some_separated_rt is None:
            some_separated_rt = 900
        if all_separated_rt is None:
            all_separated_rt = 900
        if largest_shap_rt is None:
            largest_shap_rt = 900
        if smallest_shap_rt is None:
            smallest_shap_rt = 900
        if max_range_less_than_1_rt is None:
            max_range_less_than_1_rt = 900
        if max_range_less_than_0_01_rt is None:
            max_range_less_than_0_01_rt = 900
        if max_range_less_than_0_001_rt is None:
            max_range_less_than_0_001_rt = 900
        if max_range_less_than_0_0001_rt is None:
            max_range_less_than_0_0001_rt = 900
        if max_range_less_than_0_00001_rt is None:
            max_range_less_than_0_00001_rt = 900
        if max_range_less_than_0_000001_rt is None:
            max_range_less_than_0_000001_rt = 900
        if max_range_less_than_0_0000001_rt is None:
            max_range_less_than_0_0000001_rt = 900
        if max_range_less_than_0_00000001_rt is None:
            max_range_less_than_0_00000001_rt = 900

    return {
        "dataset": dataset,
        "dataset_dim": dataset_dim,
        "timeout": timeout,
        "overall_rt": overall_rt,
        "some_separated_rt": some_separated_rt,
        "all_separated_rt": all_separated_rt,
        "largest_shap_rt": largest_shap_rt,
        "smallest_shap_rt": smallest_shap_rt,
        "max_range_less_than_1_rt": max_range_less_than_1_rt,
        "max_range_less_than_0_01_rt": max_range_less_than_0_01_rt,
        "max_range_less_than_0_001_rt": max_range_less_than_0_001_rt,
        "max_range_less_than_0_0001_rt": max_range_less_than_0_0001_rt,
        "max_range_less_than_0_00001_rt": max_range_less_than_0_00001_rt,
        "max_range_less_than_0_000001_rt": max_range_less_than_0_000001_rt,
        "max_range_less_than_0_0000001_rt": max_range_less_than_0_0000001_rt,
        "max_range_less_than_0_00000001_rt": max_range_less_than_0_00000001_rt,
    }


def _load_exactshap_run(info_path: Path) -> dict | None:
    with info_path.open("r") as f:
        info = yaml.safe_load(f)

    dataset, *_ = info_path.parent.name.split("-")

    runtime = info.get("overall", {}).get("runtime")
    timeout = True if runtime is None else False

    return {
        "dataset": dataset,
        "timeout": timeout,
        "overall_rt": runtime,
    }


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
        data[run["dataset"]]["input_dim"] = run["dataset_dim"]
        data[run["dataset"]]["bab"] = run["overall_rt"]
        data[run["dataset"]]["bab_some_separated"] = run["some_separated_rt"]
        data[run["dataset"]]["bab_all_separated"] = run["all_separated_rt"]
        data[run["dataset"]]["bab_largest_shap"] = run["largest_shap_rt"]
        data[run["dataset"]]["bab_smallest_shap"] = run["smallest_shap_rt"]
        data[run["dataset"]]["bab_max_range_less_than_1"] = run["max_range_less_than_1_rt"]
        data[run["dataset"]]["bab_max_range_less_than_0_01"] = run["max_range_less_than_0_01_rt"]
        data[run["dataset"]]["bab_max_range_less_than_0_001"] = run["max_range_less_than_0_001_rt"]
        data[run["dataset"]]["bab_max_range_less_than_0_0001"] = run["max_range_less_than_0_0001_rt"]
        data[run["dataset"]]["bab_max_range_less_than_0_00001"] = run["max_range_less_than_0_00001_rt"]
        data[run["dataset"]]["bab_max_range_less_than_0_000001"] = run["max_range_less_than_0_000001_rt"]
        data[run["dataset"]]["bab_max_range_less_than_0_0000001"] = run["max_range_less_than_0_0000001_rt"]
        data[run["dataset"]]["bab_max_range_less_than_0_00000001"] = run["max_range_less_than_0_00000001_rt"]
    for run in exactshap_runs:
        data[run["dataset"]]["exactshap"] = run["overall_rt"]

    df = pd.DataFrame.from_dict(data, orient="index")
    df = df.sort_values(by="input_dim")

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
