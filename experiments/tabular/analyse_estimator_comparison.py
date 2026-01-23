# Copyright 2025 David Boetius
import argparse
from pathlib import Path
from statistics import NormalDist, median

import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib import pyplot as plt


def _find_bab_repetition_dirs(bab_network_dir: Path) -> list[Path]:
    """Find repetition directories (e.g., repeatition_1, repeatition_2, etc.)."""
    if not bab_network_dir.exists():
        return []
    # Look for repeatition_* or repetition_* directories
    rep_dirs = []
    for pattern in ["repeatition_*", "repetition_*"]:
        rep_dirs.extend(bab_network_dir.glob(pattern))
    # Sort by directory name to ensure consistent ordering
    return sorted(rep_dirs)


def _get_bab_file_path(run_dir: Path, network: str, filename: str) -> Path | None:
    """Get path to a BaB file, handling both old format and repetition format.

    Returns the first repetition's file path if repetitions exist,
    otherwise returns the direct path if it exists, or None if neither exists.
    """
    bab_network_dir = run_dir / "BaB" / network
    direct_path = bab_network_dir / filename

    # Check if old format (direct file) exists
    if direct_path.exists():
        return direct_path

    # Check for repetition directories
    rep_dirs = _find_bab_repetition_dirs(bab_network_dir)
    if rep_dirs:
        # Use the first repetition
        first_rep_path = rep_dirs[0] / filename
        if first_rep_path.exists():
            return first_rep_path

    return None


def _load_bab_final_bounds(
    run_dir: Path, network: str
) -> tuple[dict[object, tuple[float, float]], dict[object, float], float]:
    bounds_path = _get_bab_file_path(run_dir, network, "multi_shap_bab_bounds.feather")
    if bounds_path is None:
        raise SystemExit(f"Missing BaB bounds for {network}")
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
        true_value = midpoint if np.isclose(lb, ub, atol=1e-4, rtol=1e-4) else np.nan
        bounds[feature_key] = (lb, ub)
        true_values[feature_key] = true_value
        try:
            feature_idx = int(feature_key)
            bounds[feature_idx] = (lb, ub)
            true_values[feature_idx] = true_value
        except (TypeError, ValueError):
            continue
    return bounds, true_values, max_half_width


def _load_output_scale(run_dir: Path, network: str) -> float:
    info_path = _get_bab_file_path(run_dir, network, "info.yaml")
    if info_path is None:
        raise SystemExit(f"Missing BaB info.yaml for {network}")
    with info_path.open("r") as handle:
        info = yaml.safe_load(handle)
    model_output = (
        info.get("config", {}).get("further_stats", {}).get("model_output", None)
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
    bab_network_dir = run_dir / "BaB" / network
    rep_dirs = _find_bab_repetition_dirs(bab_network_dir)

    # Check if we have repetitions or old format
    if rep_dirs:
        # New format with repetitions
        all_exact = True
        runtimes = []

        for rep_dir in rep_dirs:
            info_path = rep_dir / "info.yaml"
            bounds_path = rep_dir / "multi_shap_bab_bounds.feather"

            if not info_path.exists() or not bounds_path.exists():
                all_exact = False
                continue

            with info_path.open("r") as handle:
                info = yaml.safe_load(handle)

            max_iters_reached = info.get("overall", {}).get("max_iters", False)
            timeout_reached = info.get("overall", {}).get("timeout", False)
            is_exact = not max_iters_reached and not timeout_reached

            if not is_exact:
                all_exact = False
                continue

            bounds = pd.read_feather(bounds_path)
            if bounds.empty or "runtime" not in bounds.columns:
                all_exact = False
                continue

            runtimes.append(float(bounds["runtime"].iloc[-1]))

        if not all_exact or not runtimes:
            return False, None

        # Return median runtime across repetitions
        return True, median(runtimes)
    else:
        # Old format without repetitions
        info_path = bab_network_dir / "info.yaml"
        bounds_path = bab_network_dir / "multi_shap_bab_bounds.feather"

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
    bab_network_dir = run_dir / "BaB" / network
    rep_dirs = _find_bab_repetition_dirs(bab_network_dir)

    # Helper function to compute runtimes for a single bounds file
    def compute_runtimes_single(bounds_path: Path) -> dict[float, float | None]:
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

    if rep_dirs:
        # New format with repetitions - compute median runtime across repetitions
        all_runtimes = {threshold: [] for threshold in thresholds}

        for rep_dir in rep_dirs:
            bounds_path = rep_dir / "multi_shap_bab_bounds.feather"
            rep_runtimes = compute_runtimes_single(bounds_path)
            for threshold, runtime in rep_runtimes.items():
                if runtime is not None:
                    all_runtimes[threshold].append(runtime)

        # Compute median for each threshold
        result = {}
        for threshold in thresholds:
            if all_runtimes[threshold]:
                result[threshold] = median(all_runtimes[threshold])
            else:
                result[threshold] = None
        return result
    else:
        # Old format without repetitions
        bounds_path = bab_network_dir / "multi_shap_bab_bounds.feather"
        return compute_runtimes_single(bounds_path)


def _compute_output_ticks(output_scale: float, max_error: float) -> list[float]:
    if output_scale <= 0:
        return []
    ticks = [output_scale * 0.1, output_scale * 0.01]
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
) -> pd.Series:
    feature_cols = [col for col in df.columns if col in bounds]
    if not feature_cols:
        raise SystemExit("No overlapping feature columns found for error aggregation.")
    values = df[feature_cols].to_numpy()
    true_vals = np.array([true_values[col] for col in feature_cols])
    use_bounds = np.isnan(true_vals).any()
    if use_bounds:
        lower = np.array([bounds[col][0] for col in feature_cols])
        upper = np.array([bounds[col][1] for col in feature_cols])
        # Optimistic error calculation
        errors = np.where(
            values < lower,
            lower - values,
            np.where(values > upper, values - upper, 0.0),
        )
    else:
        errors = np.abs(values - true_vals)
    if agg_mode == "l1":
        return pd.Series(errors.mean(axis=1), index=df.index)
    if agg_mode == "l2":
        return pd.Series(np.sqrt(np.square(errors).mean(axis=1)), index=df.index)
    if agg_mode == "linf":
        return pd.Series(errors.max(axis=1), index=df.index)
    raise ValueError(f"Unknown aggregation mode: {agg_mode}")


def _load_sampling_errors(
    run_dir: Path,
    estimator: str,
    network: str,
    true_values: dict[object, float],
    bounds: dict[object, tuple[float, float]],
    agg_mode: str,
) -> pd.DataFrame:
    records = []
    sampling_root = run_dir / estimator / network
    if not sampling_root.exists():
        return pd.DataFrame()
    for stats_path in sampling_root.rglob("estimate_iter_stats.feather"):
        df = pd.read_feather(stats_path)
        errors = _compute_aggregated_error(df, true_values, bounds, agg_mode)
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
            # Extract num_samples and seed from path
            # Handle both old format (estimator/network/seed/num_samples/) and
            # new format with repetitions (estimator/network/repetition/seed/num_samples/)
            parent = stats_path.parent
            num_samples = int(parent.name)

            # Walk up the path to find seed
            seed = None
            for p in stats_path.parents:
                # Skip the parent (num_samples directory)
                if p == parent:
                    continue
                # Stop at the network directory
                if p == sampling_root:
                    break
                # Check if this looks like a seed directory (not a repetition directory)
                if not (p.name.startswith("repeatition_") or p.name.startswith("repetition_")):
                    seed = p.name
                    break

            if seed is None:
                # Fallback to old behavior
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
        mean_error="mean",
        std_error="std",
        count="count",
    )
    summary["std_error"] = summary["std_error"].fillna(0.0)
    return summary.sort_values(by=["estimator", "num_samples"])


def _prepare_area_bounds(
    summary: pd.DataFrame, area_mode: str, ci_multiplier: float | None
) -> pd.DataFrame:
    summary = summary.copy()
    if area_mode == "none":
        summary["area_lower"] = np.nan
        summary["area_upper"] = np.nan
    elif area_mode == "range" or ci_multiplier is None:
        summary["area_lower"] = summary["min_error"]
        summary["area_upper"] = summary["max_error"]
    else:
        std = summary["std_error"].fillna(0.0)
        summary["area_lower"] = summary["mean_error"] - ci_multiplier * std
        summary["area_upper"] = summary["mean_error"] + ci_multiplier * std
    summary["area_lower"] = summary["area_lower"].clip(lower=0.0)
    summary["area_upper"] = summary["area_upper"].clip(lower=0.0)
    return summary


def _parse_area_argument(area_value: str) -> tuple[str, float | None, float | None]:
    value_lower = area_value.lower()
    if value_lower == "range":
        return "range", None, None
    if value_lower == "none":
        return "none", None, None
    try:
        alpha = float(area_value)
    except ValueError as exc:
        raise SystemExit(
            "--area must be 'range', 'none', or a floating point value between 0 and 1."
        ) from exc
    if not 0.0 < alpha < 1.0:
        raise SystemExit("--area floating point values must be in (0, 1).")
    multiplier = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    return "ci", alpha, multiplier


def _select_highlight_values(runs: pd.DataFrame, highlight: str | None) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()
    if highlight is None or highlight.lower() == "none":
        return pd.DataFrame()
    selected = []
    for estimator in sorted(runs["estimator"].unique()):
        estimator_runs = runs[runs["estimator"] == estimator]
        if highlight == "mean":
            mean_error = estimator_runs.groupby("num_samples", as_index=False)[
                "error"
            ].mean()
            mean_error["estimator"] = estimator
            mean_error["name"] = "mean"
            selected.append(mean_error)
        else:
            seed = highlight
            if seed not in estimator_runs["seed"].unique():
                raise SystemExit(f"Seed {seed} not found for estimator {estimator}")
            selected.append(
                estimator_runs[estimator_runs["seed"] == seed].rename(
                    columns={"seed": "name"}
                )
            )
    return pd.concat(selected, ignore_index=True)


def _sanitize_name_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _build_wide_summary(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "min_error",
        "max_error",
        "mean_error",
        "std_error",
        "count",
        "area_lower",
        "area_upper",
    ]
    base = pd.DataFrame({"num_samples": sorted(summary["num_samples"].unique())})
    for metric in metrics:
        if metric not in summary.columns:
            continue
        pivot = summary.pivot(index="num_samples", columns="estimator", values=metric)
        pivot = pivot.rename(columns=lambda est: f"{est}_{metric}")
        pivot = pivot.reset_index()
        base = base.merge(pivot, on="num_samples", how="left")
    return base


def _merge_highlight_columns(
    data: pd.DataFrame, highlight_vals: pd.DataFrame
) -> pd.DataFrame:
    if highlight_vals.empty:
        return data
    pivot = highlight_vals.pivot(index="num_samples", columns="estimator", values="error")
    pivot = pivot.rename(columns=lambda est: f"{est}_highlight_error")
    pivot = pivot.reset_index()
    return data.merge(pivot, on="num_samples", how="left")


def _dump_network_data(
    run_dir: Path,
    network: str,
    error_agg: str,
    highlight: str | None,
    summary: pd.DataFrame,
    highlight_vals: pd.DataFrame,
) -> None:
    if summary.empty and highlight_vals.empty:
        return
    highlight_label = highlight if highlight else "none"
    safe_network = _sanitize_name_component(network)
    safe_agg = _sanitize_name_component(error_agg)
    safe_highlight = _sanitize_name_component(highlight_label)
    dump_name = f"{safe_network}_{safe_agg}_{safe_highlight}.csv"
    dump_path = run_dir / dump_name
    data = _build_wide_summary(summary)
    data = _merge_highlight_columns(data, highlight_vals)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(dump_path, index=False)


def _plot_error_summary(
    ax: plt.Axes,
    error_summary: pd.DataFrame,
    highlight_vals: pd.DataFrame,
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
    y_axis_scale: str,
    area_mode: str,
    area_alpha: float | None,
    x_max: float | None = None,
    title: str | None = None,
    estimators_only: bool = False,
) -> None:
    estimator_order = list(error_summary["estimator"].unique())
    estimator_colors = dict(
        zip(estimator_order, sns.color_palette(n_colors=len(estimator_order)))
    )
    if area_mode != "none":
        area_label = (
            "range" if area_mode == "range" or area_alpha is None else f"CI α={area_alpha:g}"
        )
        for estimator, group in error_summary.groupby("estimator"):
            group = group.sort_values("num_samples")
            ax.fill_between(
                group["num_samples"],
                group["area_lower"],
                group["area_upper"],
                color=estimator_colors.get(estimator),
                alpha=0.4,
                linewidth=1.4,
                edgecolor=estimator_colors.get(estimator),
                zorder=1,
                label=f"{estimator} {area_label}",
            )
    if not highlight_vals.empty:
        for estimator, group in highlight_vals.groupby("estimator"):
            group = group.sort_values("num_samples")
            name = group["name"].iloc[0]
            ax.plot(
                group["num_samples"],
                group["error"],
                color=estimator_colors.get(estimator),
                linewidth=2.0,
                marker="o",
                zorder=2,
                label=f"{estimator} ({name})",
            )
    ax.set_xscale("log")
    ax.set_yscale("log" if y_axis_scale == "log" else "linear")
    if x_max is not None:
        # Get current x limits to preserve the minimum
        x_min_current, _ = ax.get_xlim()
        ax.set_xlim(x_min_current, x_max)
    if estimators_only:
        # When estimators_only is True, scale y-axis to the data range
        if not np.isnan(y_min) and not np.isnan(y_max):
            ax.set_ylim(y_min, y_max)
    elif ticks:
        ticks_sorted = list(ticks)
        if exact_tick is not None:
            ticks_sorted.append(exact_tick)
        ticks_sorted = sorted(set(ticks_sorted))
        ax.set_yticks(ticks_sorted)
        if not np.isnan(y_min) and not np.isnan(y_max):
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
        ax_right.set_yscale("log" if y_axis_scale == "log" else "linear")
        if not np.isnan(y_min) and not np.isnan(y_max):
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
    highlight: str | None,
    y_axis_scale: str,
    area_mode: str,
    area_alpha: float | None,
    ci_multiplier: float | None,
    dump_data: bool,
    x_max_spec: str | None = None,
    estimators_only: bool = False,
) -> plt.Figure | None:
    bounds, true_values, last_bound_half_width = _load_bab_final_bounds(
        run_dir, network
    )
    output_scale = _load_output_scale(run_dir, network)
    is_exact, exact_runtime = _load_bab_status(run_dir, network)

    # Parse x_max specification
    x_max = None
    if x_max_spec is not None:
        if x_max_spec.endswith('n'):
            # Format: "NUMBERn" means NUMBER * num_features
            try:
                multiplier = float(x_max_spec[:-1])
                num_features = len(bounds)
                x_max = multiplier * num_features
            except ValueError as exc:
                raise SystemExit(f"Invalid --xmax format: {x_max_spec}. Expected number or 'NUMBERn'.") from exc
        else:
            # Direct numerical value
            try:
                x_max = float(x_max_spec)
            except ValueError as exc:
                raise SystemExit(f"Invalid --xmax value: {x_max_spec}. Expected number or 'NUMBERn'.") from exc
    runs_list = []
    for estimator in estimators:
        runs = _load_sampling_errors(
            run_dir,
            estimator,
            network,
            true_values,
            bounds,
            error_agg,
        )
        if not runs.empty:
            runs_list.append(runs)
    if not runs_list:
        return None
    runs = pd.concat(runs_list, ignore_index=True)
    if np.isclose(runs["error"], 0.0).all():
        print(f"Skipping {network} because all errors are zero.")
        return None
    summary = _aggregate_errors(runs)
    summary = _prepare_area_bounds(summary, area_mode, ci_multiplier)
    highlight_vals = _select_highlight_values(runs, highlight)
    if dump_data:
        _dump_network_data(
            run_dir,
            network,
            error_agg,
            highlight,
            summary,
            highlight_vals,
        )
    max_error = float(summary["max_error"].max())
    ticks = _compute_output_ticks(output_scale, max_error)
    runtime_thresholds = _load_bab_runtime_thresholds(
        run_dir, network, ticks, output_scale
    )
    exact_tick = output_scale * 1e-6 if is_exact else None
    if estimators_only:
        # Scale y-axis to the actual error range
        min_error = float(summary["min_error"].min())
        if not highlight_vals.empty:
            min_error = min(min_error, float(highlight_vals["error"].min()))
            max_error = max(max_error, float(highlight_vals["error"].max()))
        # Add some padding for better visualization
        if y_axis_scale == "log":
            y_min = min_error * 0.8
            y_max = max_error * 1.2
        else:
            margin = (max_error - min_error) * 0.1
            y_min = min_error - margin
            y_max = max_error + margin
        y_min = max(y_min, 1e-12)
    else:
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
        highlight_vals,
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
        y_axis_scale,
        area_mode,
        area_alpha,
        x_max=x_max,
        title=f"Sampling error for {network}",
        estimators_only=estimators_only,
    )
    fig.tight_layout()
    return fig


def main(
    run_dir: Path,
    network: str | None,
    out_name: str,
    sampling_estimators: list[str] | None,
    all_networks: bool,
    include_bab: bool,
    error_agg: str,
    highlight: str | None,
    y_axis_scale: str,
    area_mode: str,
    area_alpha: float | None,
    ci_multiplier: float | None,
    dump_data: bool,
    x_max_spec: str | None = None,
    estimators_only: bool = False,
) -> None:
    if sampling_estimators is None or any(
        estimator.lower() == "all" for estimator in sampling_estimators
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
    networks = _collect_networks(
        run_dir, estimators, network, all_networks, include_bab
    )
    figures = []
    for net in networks:
        fig = _plot_network(
            run_dir,
            net,
            out_name,
            estimators,
            all_networks,
            error_agg,
            highlight,
            y_axis_scale,
            area_mode,
            area_alpha,
            ci_multiplier,
            dump_data,
            x_max_spec=x_max_spec,
            estimators_only=estimators_only,
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
        help="Discover networks from estimators only (BaB still used for ground truth).",
    )
    parser.add_argument(
        "--error-agg",
        type=str,
        default="l2",
        choices=["l1", "l2", "linf"],
        help="Aggregation of per-feature errors for each run.",
    )
    parser.add_argument(
        "--highlight",
        type=str,
        default="mean",
        help="The line within the error range to highlight. "
        "Either 'mean', a specific seed, or 'none' to disable the line.",
    )
    parser.add_argument(
        "--y-axis",
        type=str,
        default="lin",
        choices=["lin", "log"],
        help="Scale to use for the error axis (linear default, or log).",
    )
    parser.add_argument(
        "--area",
        type=str,
        default="range",
        help=(
            "Shaded confidence interval: 'range' for min/max, alpha in (0,1) for a "
            "Gaussian mean±z·std interval, or 'none' to hide the shaded area."
        ),
    )
    parser.add_argument(
        "--dump-data",
        action="store_true",
        help=(
            "Store per-network CSV files named <network>_<aggregation>_<highlight>.csv "
            "containing the plotted data (highlight where available, otherwise summary)."
        ),
    )
    parser.add_argument(
        "--xmax",
        type=str,
        default=None,
        help=(
            "Maximum value for the x-axis (number of samples). Can be a direct number "
            "(e.g., '1000') or 'NUMBERn' format (e.g., '64n') where NUMBER is multiplied "
            "by the number of features. For example, '64n' with 12 features sets xmax to 768."
        ),
    )
    args = parser.parse_args()
    networks_arg = args.networks
    all_networks = networks_arg.lower() == "all"
    network = None if all_networks else networks_arg
    area_mode, area_alpha, ci_multiplier = _parse_area_argument(args.area)
    main(
        args.run_dir,
        network,
        args.out_name,
        args.sampling_estimator,
        all_networks,
        not args.estimators_only,
        args.error_agg,
        args.highlight,
        args.y_axis,
        area_mode,
        area_alpha,
        ci_multiplier,
        args.dump_data,
        x_max_spec=args.xmax,
        estimators_only=args.estimators_only,
    )
