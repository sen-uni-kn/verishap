# Copyright 2025 David Boetius
from pathlib import Path
from typing import Protocol

import pandas as pd
from formalax import Box
from ruamel.yaml import YAML


class Logger(Protocol):
    def log_stats(self, function_name: str, **stats): ...

    def log_iter_stats(self, function_name: str, i: int, **stats): ...

    def log_bounds(self, function_name: str, i: int, bounds: Box, name: str = "φ"): ...


class ConsoleLogger(Logger):
    def __init__(self):
        self.last_function = None

    def _log_new_function(self, function_name: str):
        if self.last_function != function_name:
            print(f"Now starting {function_name}.")
            print("=" * 100)
            self.last_function = function_name

    def log_stats(self, function_name: str, **stats):
        self._log_new_function(function_name)
        print(", ".join(f"{k}: {v}" for k, v in stats.items()))

    def log_iter_stats(self, function_name: str, i: int, **stats):
        self._log_new_function(function_name)
        print(f"[i: {i:3d}] {stats}")

    def log_bounds(self, function_name: str, i: int, bounds: Box, name: str = "φ"):
        self._log_new_function(function_name)
        lbs, ubs = bounds.concrete
        mid, ran = (lbs + ubs) / 2, (ubs - lbs) / 2
        mid, ran = mid.tolist(), ran.tolist()
        mids = ", ".join(f"{m:8.4f}" for m in mid)
        rans = ", ".join(f"{r:8.4f}" for r in ran)
        print(f"    {name} ∈ [{mids}]")
        print("    " + " " * len(name) + f" ± [{rans}]")


class FileLogger(Logger):
    def __init__(self, directory: Path):
        self.stats = {}
        self.iter_stats = {}
        self.bounds = {}
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=False)

    def log_stats(self, function_name: str, **stats):
        if function_name in self.stats:
            prev = self.stats[function_name]
            prev = [prev] if not isinstance(prev, list) else prev
            prev.append(stats)
            self.stats[function_name] = prev
        else:
            self.stats[function_name] = stats

    def log_iter_stats(self, function_name: str, i: int, **stats):
        if function_name not in self.iter_stats:
            self.iter_stats[function_name] = {}
        self.iter_stats[function_name][i] = stats

    def log_bounds(self, function_name: str, i: int, bounds: Box, name: str = "φ"):
        if function_name not in self.bounds:
            self.bounds[function_name] = {}
        lbs, ubs = bounds.concrete
        lbs, ubs = lbs.squeeze(), ubs.squeeze()
        if lbs.ndim == 0:
            values = {"lb": lbs.item(), "ub": ubs.item()}
        else:
            values = {(i, "lb"): lb.item() for i, lb in enumerate(lbs)} | {
                (i, "ub"): ub.item() for i, ub in enumerate(ubs)
            }
        self.bounds[function_name][i] = values

    def save_files(self):
        yaml = YAML()
        stats_file = self.directory / "info.yaml"
        yaml.dump(self.stats, stats_file)
        print(f"Overall run statistics saved to {stats_file}.")

        for function_name, iter_stats in self.iter_stats.items():
            iter_stats_file = self.directory / f"{function_name}_iter_stats.yaml"
            yaml.dump(iter_stats, iter_stats_file)
            print(f"{function_name} iteration statistics saved to {iter_stats_file}.")

        for function_name, bounds in self.bounds.items():
            if len(bounds) > 0 and len(bounds.keys()) > 2:
                num_features = len(bounds.keys()) // 2
                columns = pd.MultiIndex.from_product(
                    [[i for i in range(num_features)], ["lb", "ub"]],
                    names=["feature", "bound"],
                )
                bounds = pd.DataFrame(bounds, columns=columns)
            else:
                bounds = pd.DataFrame(bounds)
            print(f"{function_name} last bounds:")
            if len(bounds) > 0 and len(bounds.keys()) > 2:
                print(bounds.iloc[-1].unstack(level=1))
            else:
                print(bounds.iloc[-1])
            out_file = self.directory / f"{function_name}_bounds.csv"
            bounds.to_csv(out_file, index=False)
            print(f"{function_name} bounds saved to {out_file}.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.save_files()
