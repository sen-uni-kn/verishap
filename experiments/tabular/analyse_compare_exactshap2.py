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
    effective_features = int(info_path.parent.name)

    timeout = info.get("overall", {}).get("runtime", {}).get("timeout", True)
    rts = info.get("overall", {}).get("runtime", {})

    if timeout:
        rts["overall"] = 1800

    return {"effective_features": effective_features, "timeout": timeout, **rts}


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
        data[dim]["bab_timeout"] = run["timeout"]
        for key, value in run.items():
            if key != "effective_features" and key != "timeout":
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
