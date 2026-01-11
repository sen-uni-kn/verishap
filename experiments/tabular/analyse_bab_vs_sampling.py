# Copyright 2025 David Boetius
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib import pyplot as plt


def _load_bab_final_bounds(
    run_dir: Path, network: str
) -> tuple[dict[object, tuple[float, float]], dict[object, float], float]:
    bounds_path = run_dir / "BaB" / network / "multi_shap_bab_bounds.feather"
    if not bounds_path.exists():
        raise SystemExit(f"Missing BaB bounds at {bounds_path}")
    df = pd.read_feather(bounds_path)
    if df.empty:
        raise SystemExit(f"No BaB bounds found in {bounds_path}")
    last_row = df.iloc[-1]
    bounds: dict[object, tuple[float, float]] = {}
    true_values: dict[object, float] = {}
    max_half_width = 0.0
    columns = df.columns
    for feature_key in df.columns.get_level_values(0).unique():
        if (feature_key, "lb") not in columns or (feature_key, "ub") not in columns:
            continue
        lb = float(last_row[(feature_key, "lb")])
        ub = float(last_row[(feature_key, "ub")])
        midpoint = (lb + ub) / 2.0
        half_width = (ub - lb) / 2.0
        max_half_width = max(max_half_width, abs(half_width))
        bounds[feature_key] = (lb, ub)
        true_values[feature_key] = midpoint
        try:
            feature_idx = int(feature_key)
            bounds[feature_idx] = (lb, ub)
            true_values[feature_idx] = midpoint
        except (TypeError, ValueError):
            continue
    return bounds, true_values, max_half_width


def _load_output_scale(run_dir: Path, network: str) -> float:
    info_path = run_dir / "BaB" / network / "info.yaml"
    if not info_path.exists():
        raise SystemExit(f"Missing BaB info.yaml at {info_path}")
    with info_path.open("r") as handle:
        info = yaml.safe_load(handle)
    model_output = info.get("config", {}).get("further_stats", {}).get(
        "model_output", None
    )
    if model_output is None:
        raise SystemExit(f"Missing model_output in {info_path}")
    if isinstance(model_output, dict):
        values = list(model_output.values())
    elif isinstance(model_output, (list, tuple)):
        values = list(model_output)
    else:
        values = [model_output]
    values = [abs(float(value)) for value in values if value is not None]
    if not values:
        raise SystemExit(f"Empty model_output in {info_path}")
    return max(values)


def _load_bab_status(run_dir: Path, network: str) -> tuple[bool, float | None]:
    info_path = run_dir / "BaB" / network / "info.yaml"
    bounds_path = run_dir / "BaB" / network / "multi_shap_bab_bounds.feather"
    if not info_path.exists() or not bounds_path.exists():
        return False, None
    with info_path.open("r") as handle:
        info = yaml.safe_load(handle)
    max_iters_reached = info.get("overall", {}).get("max_iters", False)
    timeout_reached = info.get("overall", {}).get("timeout", False)
    is_exact = not max_iters_reached and not timeout_reached
    if not is_exact:
        return False, None
    bounds = pd.read_feather(bounds_path)
    if bounds.empty or "runtime" not in bounds.columns:
        return False, None
    return True, float(bounds["runtime"].iloc[-1])


def _get_runtime_at(condition: np.ndarray, iter_times: np.ndarray) -> float | None:
    idx = np.where(condition)[0]
    if len(idx) == 0:
        return None
    return float(iter_times[idx[0]])


def _load_bab_runtime_thresholds(
    run_dir: Path,
    network: str,
    thresholds: list[float],
    output_scale: float,
) -> dict[float, float | None]:
    bounds_path = run_dir / "BaB" / network / "multi_shap_bab_bounds.feather"
    if not bounds_path.exists():
        return {}
    bounds = pd.read_feather(bounds_path)
    if "runtime" not in bounds.columns:
        return {}
    feature_keys = []
    for feature_key in bounds.columns.get_level_values(0).unique():
        if (feature_key, "lb") in bounds.columns and (
            feature_key,
            "ub",
        ) in bounds.columns:
            feature_keys.append(feature_key)
    if not feature_keys:
        return {}
    lbs = np.array([bounds[(key, "lb")] for key in feature_keys]).T
    ubs = np.array([bounds[(key, "ub")] for key in feature_keys]).T
    half_width = (ubs - lbs) / 2.0
    max_norm = (half_width / output_scale).max(axis=1)
    iter_times = bounds["runtime"].to_numpy()
    runtimes = {}
    for threshold in thresholds:
        fraction = threshold / output_scale
        runtimes[threshold] = _get_runtime_at(max_norm <= fraction, iter_times)
    return runtimes


def _compute_output_ticks(output_scale: float, max_error: float) -> list[float]:
    if output_scale <= 0:
        return []
    ticks = [output_scale, output_scale * 0.1, output_scale * 0.01]
    tick = output_scale
    while tick < max_error:
        tick *= 10.0
        ticks.append(tick)
    return sorted(set(ticks))


def _collect_networks(
    run_dir: Path,
    estimators: list[str],
    network: str | None,
    all_networks: bool,
    include_bab: bool,
) -> list[str]:
    if not all_networks:
        if network is None:
            raise SystemExit("Provide --networks <name> or set --networks all.")
        return [network]
    networks: set[str] = set()
    if include_bab:
        bab_root = run_dir / "BaB"
        if bab_root.exists():
            networks.update(path.name for path in bab_root.iterdir() if path.is_dir())
    for estimator in estimators:
        estimator_root = run_dir / estimator
        if estimator_root.exists():
            networks.update(
                path.name for path in estimator_root.iterdir() if path.is_dir()
            )
    if not networks:
        raise SystemExit(f"No networks found under {run_dir}")
    return sorted(networks)


def _resolve_out_path(
    run_dir: Path,
    out_name: str,
    network: str,
    multi_networks: bool,
    tag: str | None = None,
) -> Path:
    out_path = Path(out_name)
    suffix = out_path.suffix
    stem = out_path.stem if suffix else out_path.name
    if multi_networks:
        stem = f"{stem}_{network}"
    if tag:
        stem = f"{stem}_{tag}"
    if suffix:
        out_path = out_path.with_name(f"{stem}{suffix}")
    else:
        out_path = out_path.with_name(stem)
    return run_dir / out_path


def _compute_aggregated_error(
    df: pd.DataFrame,
    true_values: dict[object, float],
    bounds: dict[object, tuple[float, float]],
    agg_mode: str,
    use_bounds: bool,
) -> pd.Series:
    if use_bounds:
        feature_cols = [col for col in df.columns if col in bounds]
    else:
        feature_cols = [col for col in df.columns if col in true_values]
    if not feature_cols:
        raise SystemExit("No overlapping feature columns found for error aggregation.")
    values = df[feature_cols].to_numpy()
    if use_bounds:
        lower = np.array([bounds[col][0] for col in feature_cols])
        upper = np.array([bounds[col][1] for col in feature_cols])
        errors = np.where(
            values < lower,
            lower - values,
            np.where(values > upper, values - upper, 0.0),
        )
    else:
        true_vals = np.array([true_values[col] for col in feature_cols])
        errors = np.abs(values - true_vals)
    if agg_mode == "mean":
        return pd.Series(errors.mean(axis=1), index=df.index)
    if agg_mode == "max":
        return pd.Series(errors.max(axis=1), index=df.index)
    raise ValueError(f"Unknown aggregation mode: {agg_mode}")


def _load_sampling_errors(
    run_dir: Path,
    estimator: str,
    network: str,
    true_values: dict[object, float],
    bounds: dict[object, tuple[float, float]],
    agg_mode: str,
    use_bounds: bool,
) -> pd.DataFrame:
    records = []
    sampling_root = run_dir / estimator / network
    if not sampling_root.exists():
        return pd.DataFrame()
    for stats_path in sampling_root.rglob("estimate_iter_stats.feather"):
        df = pd.read_feather(stats_path)
        errors = _compute_aggregated_error(df, true_values, bounds, agg_mode, use_bounds)
        if "iter_key_0" in df.columns and "iter_key_1" in df.columns:
            sub = df[["iter_key_0", "iter_key_1"]].copy()
            sub["error"] = errors
            sub = sub.rename(
                columns={
                    "iter_key_0": "num_samples",
                    "iter_key_1": "seed",
                }
            )
            sub["estimator"] = estimator
            records.append(sub)
        else:
            num_samples = int(stats_path.parent.name)
            seed = stats_path.parents[1].name
            sub = pd.DataFrame({"error": errors})
            sub["num_samples"] = num_samples
            sub["seed"] = seed
            sub["estimator"] = estimator
            records.append(sub)
    if not records:
        return pd.DataFrame()
    combined = pd.concat(records, ignore_index=True)
    combined["num_samples"] = combined["num_samples"].astype(int)
    combined["seed"] = combined["seed"].astype(str)
    return combined


def _aggregate_errors(runs: pd.DataFrame) -> pd.DataFrame:
    grouped = runs.groupby(["estimator", "num_samples"], as_index=False)
    summary = grouped["error"].agg(
        min_error="min",
        max_error="max",
        count="count",
    )
    return summary.sort_values(by=["estimator", "num_samples"])


def _select_highlight_runs(
    runs: pd.DataFrame, highlight_seed: str | None
) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()
    selected = []
    for estimator in sorted(runs["estimator"].unique()):
        estimator_runs = runs[runs["estimator"] == estimator]
        seed = highlight_seed
        if seed is None or seed not in set(estimator_runs["seed"]):
            seed = sorted(estimator_runs["seed"].unique())[0]
        selected.append(estimator_runs[estimator_runs["seed"] == seed])
    return pd.concat(selected, ignore_index=True)


def _plot_error_summary(
    ax: plt.Axes,
    error_summary: pd.DataFrame,
    highlight_runs: pd.DataFrame,
    agg_mode: str,
    output_scale: float,
    runtime_thresholds: dict[float, float | None],
    exact_runtime: float | None,
    exact_tick: float | None,
    y_min: float,
    y_max: float,
    ticks: list[float],
    is_exact: bool,
    last_bound_half_width: float,
    title: str | None = None,
) -> None:
    estimator_order = list(error_summary["estimator"].unique())
    estimator_colors = dict(
        zip(estimator_order, sns.color_palette(n_colors=len(estimator_order)))
    )
    for estimator, group in error_summary.groupby("estimator"):
        group = group.sort_values("num_samples")
        ax.fill_between(
            group["num_samples"],
            group["min_error"],
            group["max_error"],
            color=estimator_colors.get(estimator),
            alpha=0.4,
            linewidth=1.4,
            edgecolor=estimator_colors.get(estimator),
            zorder=1,
            label=f"{estimator} range",
        )
    for estimator, group in highlight_runs.groupby("estimator"):
        group = group.sort_values("num_samples")
        seed = group["seed"].iloc[0]
        ax.plot(
            group["num_samples"],
            group["error"],
            color=estimator_colors.get(estimator),
            linewidth=2.0,
            marker="o",
            zorder=2,
            label=f"{estimator} (seed {seed})",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    if ticks:
        ticks_sorted = list(ticks)
        if exact_tick is not None:
            ticks_sorted.append(exact_tick)
        ticks_sorted = sorted(set(ticks_sorted))
        ax.set_yticks(ticks_sorted)
        ax.set_ylim(y_min, y_max)
        labels = []
        for tick_value in ticks_sorted:
            if exact_tick is not None and tick_value == exact_tick:
                labels.append("exact")
            else:
                percent = (tick_value / output_scale) * 100.0
                labels.append(f"{percent:.0f}% output")
        ax.set_yticklabels(labels)
        ax_right = ax.twinx()
        ax_right.set_yscale("log")
        ax_right.set_ylim(y_min, y_max)
        ax_right.set_yticks(ticks_sorted)
        runtime_labels = []
        for tick_value in ticks_sorted:
            if exact_tick is not None and tick_value == exact_tick:
                runtime = exact_runtime
            else:
                runtime = runtime_thresholds.get(tick_value)
            if runtime is None:
                runtime_labels.append("not reached")
            elif runtime >= 100:
                runtime_labels.append(f"{runtime:.0f}s")
            else:
                runtime_labels.append(f"{runtime:.1f}s")
        ax_right.set_yticklabels(runtime_labels)
        ax_right.set_ylabel("BaB runtime to reach error")
    if not is_exact and last_bound_half_width > 0:
        ax.axhline(
            last_bound_half_width,
            color="black",
            linestyle="--",
            linewidth=1.2,
            zorder=1,
            label="BaB last bound",
        )
    ax.set_xlabel("Number of samples")
    ax.set_ylabel(f"Aggregated error ({agg_mode})")
    if title:
        ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="best")


def _plot_network(
    run_dir: Path,
    network: str,
    out_name: str,
    estimators: list[str],
    multi_networks: bool,
    error_agg: str,
    highlight_seed: str | None,
) -> plt.Figure | None:
    bounds, true_values, last_bound_half_width = _load_bab_final_bounds(
        run_dir, network
    )
    output_scale = _load_output_scale(run_dir, network)
    is_exact, exact_runtime = _load_bab_status(run_dir, network)
    use_bounds = not is_exact
    runs_list = []
    for estimator in estimators:
        runs = _load_sampling_errors(
            run_dir,
            estimator,
            network,
            true_values,
            bounds,
            error_agg,
            use_bounds,
        )
        if not runs.empty:
            runs_list.append(runs)
    if not runs_list:
        return None
    runs = pd.concat(runs_list, ignore_index=True)
    summary = _aggregate_errors(runs)
    highlight_runs = _select_highlight_runs(runs, highlight_seed)
    max_error = float(summary["max_error"].max())
    ticks = _compute_output_ticks(output_scale, max_error)
    runtime_thresholds = _load_bab_runtime_thresholds(
        run_dir, network, ticks, output_scale
    )
    exact_tick = output_scale * 1e-6 if is_exact else None
    y_min = min(ticks)
    if exact_tick is not None:
        y_min = min(y_min, exact_tick)
    if not is_exact and last_bound_half_width > 0:
        y_min = min(y_min, last_bound_half_width)
    y_min = max(y_min, 1e-12)
    y_max = max(max_error, max(ticks))
    if exact_tick is not None:
        y_max = max(y_max, exact_tick)
    if not is_exact and last_bound_half_width > 0:
        y_max = max(y_max, last_bound_half_width)
    out_csv = _resolve_out_path(run_dir, out_name, network, multi_networks)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not summary.empty:
        summary.to_csv(out_csv, index=False)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_error_summary(
        ax,
        summary,
        highlight_runs,
        error_agg,
        output_scale,
        runtime_thresholds,
        exact_runtime,
        exact_tick,
        y_min,
        y_max,
        ticks,
        is_exact,
        last_bound_half_width,
        title=f"Sampling error vs. BaB midpoint for {network}",
    )
    fig.tight_layout()
    return fig


def _plot_network_all_features(
    run_dir: Path,
    network: str,
    out_name: str,
    estimators: list[str],
    multi_networks: bool,
    error_agg: str,
    highlight_seed: str | None,
) -> plt.Figure | None:
    return _plot_network(
        run_dir,
        network,
        out_name,
        estimators,
        multi_networks,
        error_agg,
        highlight_seed,
    )


def main(
    run_dir: Path,
    network: str | None,
    feature: int | None,
    out_name: str,
    sampling_estimators: list[str] | None,
    all_networks: bool,
    include_bab: bool,
    error_agg: str,
    highlight_seed: str | None,
) -> None:
    if feature is not None:
        raise SystemExit(
            "This analysis aggregates errors across all features. Use --feature all."
        )
    if (
        sampling_estimators is None
        or any(estimator.lower() == "all" for estimator in sampling_estimators)
    ):
        estimators = sorted(
            path.name
            for path in run_dir.iterdir()
            if path.is_dir() and path.name not in {"BaB", "warmup"}
        )
    else:
        estimators = sampling_estimators
    if not estimators:
        raise SystemExit(f"No sampling estimators found under {run_dir}")
    networks = _collect_networks(run_dir, estimators, network, all_networks, include_bab)
    figures = []
    for net in networks:
        if feature is None:
            fig = _plot_network_all_features(
                run_dir,
                net,
                out_name,
                estimators,
                all_networks,
                error_agg,
                highlight_seed,
            )
        else:
            fig = _plot_network(
                run_dir,
                net,
                out_name,
                estimators,
                all_networks,
                error_agg,
                highlight_seed,
            )
        if fig is not None:
            figures.append(fig)
    if figures:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Analyse bab_vs_sampling outputs and plot aggregated estimation errors vs "
            "sample count."
        )
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Experiment output directory (e.g., output/bab_vs_sampling/<run>).",
    )
    parser.add_argument(
        "--networks",
        type=str,
        required=True,
        help="Network name inside the run directory, or 'all'.",
    )
    parser.add_argument(
        "--feature",
        type=str,
        required=True,
        help="Must be 'all' (errors aggregate across all features).",
    )
    parser.add_argument(
        "--out-name",
        type=str,
        default="bab_vs_sampling_plot.csv",
        help="Filename for the aggregated data (saved inside run_dir).",
    )
    parser.add_argument(
        "--sampling-estimator",
        type=str,
        nargs="+",
        default=["PermutationSHAP"],
        help="Sampling estimators to plot, or include 'all' for every estimator.",
    )
    parser.add_argument(
        "--estimators-only",
        action="store_true",
        help="Discover networks from estimators only (BaB still used for truth).",
    )
    parser.add_argument(
        "--error-agg",
        type=str,
        default="mean",
        choices=["mean", "max"],
        help="Aggregation of per-feature errors for each run.",
    )
    parser.add_argument(
        "--highlight-seed",
        type=str,
        default=None,
        help="Seed to highlight for each estimator (falls back to first if missing).",
    )
    args = parser.parse_args()
    networks_arg = args.networks
    all_networks = networks_arg.lower() == "all"
    network = None if all_networks else networks_arg
    feature_arg = args.feature
    feature = None if feature_arg.lower() == "all" else int(feature_arg)
    main(
        args.run_dir,
        network,
        feature,
        args.out_name,
        args.sampling_estimator,
        all_networks,
        not args.estimators_only,
        args.error_agg,
        args.highlight_seed,
    )
