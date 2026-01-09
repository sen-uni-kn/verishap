# Copyright 2025 David Boetius
from collections import defaultdict
from pathlib import Path
from typing import Protocol

import pandas as pd
from formalax import Box
from ruamel.yaml import YAML, representer

representer.SafeRepresenter.add_representer(
    defaultdict, representer.SafeRepresenter.represent_dict
)


class Logger(Protocol):
    def log_config(self, config_name: str, config1: dict | None = None, **config2): ...

    def log_stats(
        self,
        function_name: str,
        stats1: dict | None = None,
        temporary: bool = False,
        **stats2,
    ): ...

    def log_iter_stats(
        self, function_name: str, i: int | tuple[int, ...], stats1: dict | None = None, **stats2
    ): ...

    def log_bounds(
        self,
        function_name: str,
        i: int,
        bounds: Box | tuple,
        name: str = "φ",
        runtime: float | None = None,
    ): ...


class ConsoleLogger(Logger):
    def __init__(self):
        self.last_function = None

    def _log_new_function(self, function_name: str):
        if self.last_function != function_name:
            print(f"Now starting {function_name}.")
            print("=" * 100)
            self.last_function = function_name

    def log_config(self, config_name: str, config1: dict | None = None, **config2):
        print(config_name)
        config = config2 if config1 is None else config1 | config2
        print(", ".join(f"{k}: {v}" for k, v in config.items()))

    def log_stats(
        self,
        function_name: str,
        stats1: dict | None = None,
        temporary: bool = False,
        **stats2,
    ):
        if not temporary:
            self._log_new_function(function_name)
            stats = stats2 if stats1 is None else stats1 | stats2
            print(", ".join(f"{k}: {v}" for k, v in stats.items()))

    def log_iter_stats(
        self, function_name: str, i: int | tuple[int, ...], stats1: dict | None = None, **stats2
    ):
        self._log_new_function(function_name)
        stats = stats2 if stats1 is None else stats1 | stats2
        stats = ", ".join(f"{k}: {v}" for k, v in stats.items())
        print(f"[i: {'/'.join(f'{j:3d}' for j in i)}] {stats}")

    def log_bounds(
        self,
        function_name: str,
        i: int,
        bounds: Box | tuple,
        name: str = "φ",
        runtime: float | None = None,
    ):
        self._log_new_function(function_name)
        lbs, ubs = bounds
        mid, ran = (lbs + ubs) / 2, (ubs - lbs) / 2
        mid, ran = mid.squeeze(), ran.squeeze()
        if mid.ndim > 0:
            mid, ran = mid.tolist(), ran.tolist()
            mids = ", ".join(f"{m:8.4f}" for m in mid)
            rans = ", ".join(f"{r:8.4f}" for r in ran)
            print(f"    {name} ∈ [{mids}]")
            print("    " + " " * len(name) + f" ± [{rans}]")
        else:
            mid, ran = mid.item(), ran.item()
            print(f"    {name} ∈ [{mid:8.4f} ± {ran:8.4f}]")


class FileLogger(Logger):
    def __init__(self, directory: Path):
        self.info = {"config": {}}
        self.iter_stats = {}
        self.bounds = {}
        self.bound_runtimes = {}
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def log_config(self, config_name: str, config1: dict | None = None, **config2):
        config = config2 if config1 is None else config1 | config2
        self.info["config"][config_name] = config

    def log_stats(
        self,
        function_name: str,
        stats1: dict | None = None,
        temporary: bool = False,
        **stats2,
    ):
        stats = stats2 if stats1 is None else stats1 | stats2
        self.info[function_name] = stats

    def log_iter_stats(
        self, function_name: str, i: int | tuple[int, ...], stats1: dict | None = None, **stats2
    ):
        stats = stats2 if stats1 is None else stats1 | stats2
        if function_name not in self.iter_stats:
            self.iter_stats[function_name] = []
        if isinstance(i, int):
            self.iter_stats[function_name].append({"iteration": i} | stats)
        else:
            self.iter_stats[function_name].append({f"iteration_{j}": j for j in i} | stats)


    def log_bounds(
        self,
        function_name: str,
        i: int,
        bounds: Box | tuple,
        name: str = "φ",
        runtime: float | None = None,
    ):
        if function_name not in self.bounds:
            self.bounds[function_name] = []
            self.bound_runtimes[function_name] = []
        lbs, ubs = bounds
        lbs, ubs = lbs.squeeze(), ubs.squeeze()
        if lbs.ndim == 0:
            values = {"lb": lbs.item(), "ub": ubs.item()}
        else:
            values = {(i, "lb"): lb.item() for i, lb in enumerate(lbs)} | {
                (i, "ub"): ub.item() for i, ub in enumerate(ubs)
            }
        self.bounds[function_name].append(values)
        self.bound_runtimes[function_name].append(runtime)

    def save_files(self):
        yaml = YAML(typ="safe")
        stats_file = self.directory / "info.yaml"
        yaml.dump(self.info, stats_file)
        print(f"Overall run statistics saved to {stats_file}.")

        for function_name, iter_stats in self.iter_stats.items():
            iter_stats_file = self.directory / f"{function_name}_iter_stats.feather"
            iter_stats = pd.DataFrame(iter_stats)
            iter_stats.to_feather(iter_stats_file, compression="zstd", compression_level=9)
            print(f"{function_name} iteration statistics saved to {iter_stats_file}.")

        for function_name, bounds in self.bounds.items():
            print(f"{function_name} last bounds:")
            if len(bounds) > 0 and len(bounds[0].keys()) > 2:
                num_features = len(bounds[0].keys()) // 2
                columns = pd.MultiIndex.from_product(
                    [[i for i in range(num_features)], ["lb", "ub"]],
                    names=["feature", "bound"],
                )
                bounds = pd.DataFrame(bounds, columns=columns)
                print(bounds.iloc[-1].unstack(level=1))
            else:
                bounds = pd.DataFrame(bounds)
                print(bounds.iloc[-1])

            bounds["runtime"] = self.bound_runtimes[function_name]

            out_file = self.directory / f"{function_name}_bounds.feather"
            bounds.to_feather(out_file, compression="zstd", compression_level=9)
            print(f"{function_name} bounds saved to {out_file}.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.save_files()


class JoinLoggers(Logger):
    def __init__(self, *loggers: Logger):
        self.loggers = loggers

    def log_config(self, config_name: str, config1: dict | None = None, **config2):
        for logger in self.loggers:
            logger.log_config(config_name, config1, **config2)

    def log_stats(
        self,
        function_name: str,
        stats1: dict | None = None,
        temporary: bool = False,
        **stats2,
    ):
        for logger in self.loggers:
            logger.log_stats(function_name, stats1, temporary=temporary, **stats2)

    def log_iter_stats(
        self, function_name: str, i: int | tuple[int, ...], stats1: dict | None = None, **stats2
    ):
        for logger in self.loggers:
            logger.log_iter_stats(function_name, i, stats1, **stats2)

    def log_bounds(
        self,
        function_name: str,
        i: int,
        bounds: Box | tuple,
        name: str = "φ",
        runtime: float | None = None,
    ):
        for logger in self.loggers:
            logger.log_bounds(function_name, i, bounds, name, runtime)


class Silence(Logger):
    """Drops all log messages."""

    def log_config(self, config_name: str, config1: dict | None = None, **config2):
        pass

    def log_stats(
        self,
        function_name: str,
        stats1: dict | None = None,
        temporary: bool = False,
        **stats2,
    ):
        pass

    def log_iter_stats(
        self, function_name: str, i: int, stats1: dict | None = None, **stats2
    ):
        pass

    def log_bounds(
        self,
        function_name: str,
        i: int,
        bounds: Box | tuple,
        name: str = "φ",
        runtime: float | None = None,
    ):
        pass
