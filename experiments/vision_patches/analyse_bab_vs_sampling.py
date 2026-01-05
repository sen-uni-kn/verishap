# Copyright 2025 David Boetius
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def _load_sampling_runs(run_dir: Path, estimator: str, feature: int) -> pd.DataFrame:
    records: list[dict] = []
    for stats_path in (run_dir / estimator).rglob(
        "seed_*/*/estimate_iter_stats.feather"
    ):
        num_samples = int(stats_path.parent.name)
        seed = stats_path.parents[1].name
        df = pd.read_feather(stats_path)
        if feature in df.columns:
            feature_col = feature
        elif str(feature) in df.columns:
            feature_col = str(feature)
        else:
            raise ValueError(f"Feature {feature} not found in {stats_path}")
        for _, row in df.iterrows():
            records.append(
                {
                    "estimator": estimator,
                    "seed": seed,
                    "num_samples": num_samples,
                    "importance": float(row[feature_col]),
                }
            )
    if not records:
        raise SystemExit(f"No sampling runs found under {run_dir}")
    return pd.DataFrame.from_records(records)


def _aggregate_runs(runs: pd.DataFrame, conf_level: float) -> pd.DataFrame:
    alpha = 1.0 - conf_level
    lower_q = alpha / 2.0
    upper_q = 1.0 - (alpha / 2.0)
    grouped = runs.groupby(["estimator", "num_samples"], as_index=False)
    summary = grouped["importance"].agg(
        mean="mean",
        lower=lambda s: s.quantile(lower_q),
        upper=lambda s: s.quantile(upper_q),
        count="count",
    )
    return summary.sort_values(by=["estimator", "num_samples"])


def _load_bab_bounds(
    run_dir: Path, feature: int, bab_batch_size: int
) -> pd.DataFrame | None:
    bounds_path = run_dir / "BaB" / "multi_shap_bab_bounds.feather"
    if not bounds_path.exists():
        return None
    df = pd.read_feather(bounds_path)
    feature_key = str(feature)
    if feature_key not in df.columns.get_level_values(0):
        raise ValueError(f"Feature {feature} not found in {bounds_path}")
    records = []
    for iteration, row in df.iterrows():
        lb = float(row[(feature_key, "lb")])
        ub = float(row[(feature_key, "ub")])
        records.append(
            {
                "estimator": "BaB",
                "num_samples": 2 * bab_batch_size * int(iteration),
                "mean": (lb + ub) / 2.0,
                "lower": lb,
                "upper": ub,
                "count": 1,
            }
        )
    return pd.DataFrame.from_records(records)


def main(
    run_dir: Path,
    feature: int,
    conf_level: float,
    bab_batch_size: int,
    out_name: str,
    sampling_estimator: str,
) -> None:
    runs = _load_sampling_runs(run_dir, sampling_estimator, feature)
    summary = _aggregate_runs(runs, conf_level)
    sampling_summary = summary[summary["estimator"] != "BaB"]
    if sampling_estimator is not None:
        sampling_summary = sampling_summary[
            sampling_summary["estimator"] == sampling_estimator
        ]
        if sampling_summary.empty:
            raise SystemExit(
                f"No sampling runs found for estimator '{sampling_estimator}'."
            )
    bab_summary = _load_bab_bounds(run_dir, feature, bab_batch_size)
    if bab_summary is not None:
        summary = pd.concat([summary, bab_summary], ignore_index=True)

    out_csv = run_dir / out_name
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    if bab_summary is not None:
        ax.plot(
            bab_summary["num_samples"],
            bab_summary["lower"],
            color="green",
            linewidth=1.2,
            zorder=1,
            label="_nolegend_",
        )
        ax.plot(
            bab_summary["num_samples"],
            bab_summary["upper"],
            color="green",
            linewidth=1.2,
            zorder=1,
            label="BaB bounds",
        )
        ax.fill_between(
            bab_summary["num_samples"],
            bab_summary["lower"],
            bab_summary["upper"],
            color="green",
            alpha=0.15,
            zorder=1,
            label="_nolegend_",
        )
    estimator_order = list(sampling_summary["estimator"].unique())
    estimator_colors = dict(
        zip(estimator_order, sns.color_palette(n_colors=len(estimator_order)))
    )
    sns.lineplot(
        data=sampling_summary,
        x="num_samples",
        y="mean",
        hue="estimator",
        hue_order=estimator_order,
        palette=estimator_colors,
        marker="o",
        ax=ax,
        zorder=3,
    )
    for estimator, group in sampling_summary.groupby("estimator"):
        group = group.sort_values("num_samples")
        ax.fill_between(
            group["num_samples"],
            group["lower"],
            group["upper"],
            color=estimator_colors.get(estimator),
            alpha=0.5,
            linewidth=1.6,
            edgecolor=estimator_colors.get(estimator),
            zorder=2,
            label="_nolegend_",
        )
    y_min = float(sampling_summary["lower"].min())
    y_max = float(sampling_summary["upper"].max())
    padding = (y_max - y_min) * 0.05
    if padding == 0:
        padding = max(1e-6, abs(y_max) * 0.05)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_xlabel("Number of samples")
    ax.set_ylabel(f"Feature {feature} importance")
    ax.set_title("Sampling importance vs. samples")
    ax.legend(loc="best")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse bab_vs_sampling outputs and plot feature importance vs samples."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Experiment output directory (e.g., output/bab_vs_sampling/<run>).",
    )
    parser.add_argument(
        "--feature",
        type=int,
        required=True,
        help="Feature index to plot.",
    )
    parser.add_argument(
        "--conf-level",
        type=float,
        default=0.99,
        help="Empirical confidence level for bounds.",
    )
    parser.add_argument(
        "--bab-batch-size",
        type=int,
        default=4096,
        help="BaB batch size per iteration (default: 4096).",
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
        default="PermutationSHAP",
        help="If set, plot only this sampling estimator (e.g., KernelSHAP).",
    )
    args = parser.parse_args()
    main(
        args.run_dir,
        args.feature,
        args.conf_level,
        args.bab_batch_size,
        args.out_name,
        args.sampling_estimator,
    )
