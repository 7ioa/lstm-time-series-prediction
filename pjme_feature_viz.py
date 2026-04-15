#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exploratory visualization for PJME hourly load dataset.

This script focuses on time-series characteristics that are important for
power-load data: trend, daily/weekly seasonality, yearly seasonality, and
autocorrelation.

Example:
  python pjme_feature_viz.py --csv PJME_hourly.csv --outdir runs/pjme_eda --width 13 --height 9 --heatmap-year 2015
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature visualization for PJME hourly load.")
    parser.add_argument("--csv", type=Path, default=Path("PJME_hourly.csv"), help="Input CSV path")
    parser.add_argument("--datetime-col", type=str, default="Datetime", help="Datetime column name")
    parser.add_argument("--value-col", type=str, default="PJME_MW", help="Value column name")
    parser.add_argument("--outdir", type=Path, default=Path("runs/pjme_eda"), help="Output directory")
    parser.add_argument("--width", type=float, default=13.0, help="Figure width in inches")
    parser.add_argument("--height", type=float, default=9.0, help="Figure height in inches")
    parser.add_argument("--dpi", type=int, default=150, help="Figure dpi")
    parser.add_argument("--heatmap-year", type=int, default=None, help="Specific year for day-hour heatmap")
    parser.add_argument("--max-lag", type=int, default=24 * 14, help="Max lag for autocorrelation plot")
    return parser.parse_args()


def apply_style() -> None:
    plt.style.use("seaborn-v0_8-white")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        }
    )


def load_series(args: argparse.Namespace) -> pd.Series:
    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    if args.datetime_col not in df.columns or args.value_col not in df.columns:
        raise ValueError(f"Missing columns. Available: {list(df.columns)}")

    df[args.datetime_col] = pd.to_datetime(df[args.datetime_col], errors="coerce")
    df = df.dropna(subset=[args.datetime_col, args.value_col])

    # Average duplicated timestamps if any.
    s = df.groupby(args.datetime_col, as_index=True)[args.value_col].mean().sort_index().astype(float)
    return s


def get_summary_stats(s: pd.Series) -> dict:
    full_index = pd.date_range(s.index.min(), s.index.max(), freq="h")
    missing = len(full_index.difference(s.index))

    hour_mean = s.groupby(s.index.hour).mean()
    dow_mean = s.groupby(s.index.dayofweek).mean()
    month_mean = s.groupby(s.index.month).mean()

    stats = {
        "rows": len(s),
        "start": s.index.min(),
        "end": s.index.max(),
        "missing": missing,
        "mean": s.mean(),
        "std": s.std(),
        "min": s.min(),
        "max": s.max(),
        "acf_1": s.autocorr(lag=1),
        "acf_24": s.autocorr(lag=24),
        "acf_168": s.autocorr(lag=168),
        "acf_720": s.autocorr(lag=24 * 30),
        "peak_hour": int(hour_mean.idxmax()),
        "valley_hour": int(hour_mean.idxmin()),
        "peak_dow": int(dow_mean.idxmax()),
        "valley_dow": int(dow_mean.idxmin()),
        "peak_month": int(month_mean.idxmax()),
        "valley_month": int(month_mean.idxmin()),
        "hour_mean": hour_mean,
        "dow_mean": dow_mean.reindex(range(7)),
        "month_mean": month_mean.reindex(range(1, 13)),
    }
    return stats


def print_summary(stats: dict) -> None:
    print("=== PJME Dataset Summary ===")
    print(f"Rows (after dedup by mean): {stats['rows']}")
    print(f"Time range: {stats['start']} -> {stats['end']}")
    print(f"Missing hourly timestamps in full range: {stats['missing']}")
    print(f"Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}, Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")
    print(f"Autocorr(lag=   1): {stats['acf_1']:.4f}")
    print(f"Autocorr(lag=  24): {stats['acf_24']:.4f}")
    print(f"Autocorr(lag= 168): {stats['acf_168']:.4f}")
    print(f"Autocorr(lag= 720): {stats['acf_720']:.4f}")
    print(f"Peak hour (mean): {stats['peak_hour']}:00")
    print(f"Valley hour (mean): {stats['valley_hour']}:00")
    print(f"Peak weekday (0=Mon): {stats['peak_dow']}")
    print(f"Valley weekday (0=Mon): {stats['valley_dow']}")
    print(f"Peak month: {stats['peak_month']}")
    print(f"Valley month: {stats['valley_month']}")


def _beautify_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.tick_params(direction="out", length=4, width=0.8, which="both")


def save_overview(s: pd.Series, outdir: Path, width: float, height: float, dpi: int, stats: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(width, height), dpi=dpi, constrained_layout=True)

    daily = s.resample("D").mean()
    axes[0, 0].plot(daily.index, daily.values, color="#2E4164", linewidth=0.8, alpha=0.8)
    axes[0, 0].plot(
        daily.index,
        daily.rolling(30, min_periods=1).mean().values,
        color="#800000",
        linewidth=1.5,
    )
    axes[0, 0].set_title("(a) Long-Term Trend", fontweight="bold")
    axes[0, 0].set_xlabel("Date")
    axes[0, 0].set_ylabel("Load (MW)")

    hour_stats = s.groupby(s.index.hour).agg(["mean", lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)])
    hour_stats.columns = ["mean", "q25", "q75"]
    hrs = np.arange(24)
    axes[0, 1].plot(hrs, hour_stats["mean"], color="#2E4164", linewidth=1.8)
    axes[0, 1].fill_between(hrs, hour_stats["q25"], hour_stats["q75"], color="#5E7994", alpha=0.3)
    axes[0, 1].scatter(stats["peak_hour"], hour_stats.loc[stats["peak_hour"], "mean"], color="#800000", s=40, zorder=5, marker="o", edgecolors="white", linewidth=0.5)
    axes[0, 1].scatter(stats["valley_hour"], hour_stats.loc[stats["valley_hour"], "mean"], color="#006400", s=40, zorder=5, marker="o", edgecolors="white", linewidth=0.5)
    axes[0, 1].set_title("(b) Daily Cycle by Hour", fontweight="bold")
    axes[0, 1].set_xlabel("Hour of Day")
    axes[0, 1].set_ylabel("Load (MW)")
    axes[0, 1].set_xticks(np.arange(0, 24, 4))

    dow_mean = stats["dow_mean"]
    bars = axes[1, 0].bar(np.arange(7), dow_mean.values, color="#2E4164", edgecolor="black", linewidth=0.8)
    axes[1, 0].set_title("(c) Weekly Cycle", fontweight="bold")
    axes[1, 0].set_xlabel("Day of Week")
    axes[1, 0].set_ylabel("Load (MW)")
    axes[1, 0].set_xticks(np.arange(7))
    axes[1, 0].set_xticklabels(DOW_LABELS)

    month_mean = stats["month_mean"]
    axes[1, 1].plot(np.arange(1, 13), month_mean.values, color="#006400", marker="o", markersize=6, linewidth=2, markerfacecolor="#2E4164", markeredgewidth=0.5, markeredgecolor="black")
    axes[1, 1].set_title("(d) Yearly Seasonality", fontweight="bold")
    axes[1, 1].set_xlabel("Month")
    axes[1, 1].set_ylabel("Load (MW)")
    axes[1, 1].set_xticks(np.arange(1, 13))
    axes[1, 1].set_xticklabels(MONTH_LABELS, rotation=0)

    for ax in axes.flatten():
        _beautify_axis(ax)

    out = outdir / "pjme_feature_overview.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def save_acf_plot(s: pd.Series, outdir: Path, width: float, height: float, dpi: int, max_lag: int) -> None:
    max_lag = int(min(max_lag, len(s) - 2))
    lags = np.arange(1, max_lag + 1)
    acf_vals = np.array([s.autocorr(lag=int(lag)) for lag in lags], dtype=float)

    fig, ax = plt.subplots(figsize=(width, max(4.5, height * 0.45)), dpi=dpi)
    ax.plot(lags, acf_vals, color="#2E4164", linewidth=1.2)
    ax.axhline(0.0, color="#444444", linewidth=0.8, linestyle="-")

    for special_lag, color in [(24, "#800000"), (168, "#006400")]:
        if special_lag <= max_lag:
            ax.axvline(special_lag, color=color, linestyle="--", linewidth=1.0)
            val = s.autocorr(lag=special_lag)
            ax.annotate(f"{special_lag}", xy=(special_lag, 0.05), fontsize=10, color=color, ha="center")

    ax.set_title("Autocorrelation vs. Lag", fontweight="bold")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("ACF")
    ax.set_xlim(1, max_lag)
    ax.set_ylim(-0.2, 1.02)
    _beautify_axis(ax)
    fig.tight_layout()

    out = outdir / "pjme_acf.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def save_year_heatmap(s: pd.Series, outdir: Path, width: float, height: float, dpi: int, year: int | None) -> None:
    target_year = year
    if target_year is None:
        year_counts = s.index.year.value_counts()
        target_year = int(year_counts.index[0])

    s_year = s[s.index.year == target_year]
    if s_year.empty:
        print(f"Skip heatmap: no data in year {target_year}")
        return

    temp = s_year.to_frame("load")
    temp["date"] = temp.index.date
    temp["hour"] = temp.index.hour
    pivot = temp.pivot_table(index="hour", columns="date", values="load", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(width, max(4.6, height * 0.5)), dpi=dpi)
    im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="YlOrRd")

    ax.set_title(f"Day-Hour Heatmap ({target_year})")
    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Hour of Day")
    ax.set_yticks(np.arange(0, 24, 2))

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Load (MW)")
    fig.tight_layout()

    out = outdir / f"pjme_heatmap_{target_year}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def save_seasonality_figure(s: pd.Series, outdir: Path, width: float, height: float, dpi: int, stats: dict) -> None:
    """Dedicated seasonal analysis figure: month distribution + month-hour map."""
    df = s.to_frame("load")
    df["month"] = df.index.month
    df["hour"] = df.index.hour

    fig, axes = plt.subplots(
        2, 1, figsize=(width, max(7.5, height * 0.95)), dpi=dpi, constrained_layout=True
    )

    # (a) Monthly distribution (boxplot) to show seasonal level + spread.
    month_data = [df.loc[df["month"] == m, "load"].values for m in range(1, 13)]
    box = axes[0].boxplot(
        month_data,
        patch_artist=True,
        widths=0.65,
        showfliers=False,
        medianprops=dict(color="#800000", linewidth=1.2),
    )
    for patch in box["boxes"]:
        patch.set_facecolor("#2E4164")
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)
    axes[0].set_title("(a) Seasonal Distribution by Month", fontweight="bold")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Load (MW)")
    axes[0].set_xticks(np.arange(1, 13))
    axes[0].set_xticklabels(MONTH_LABELS)
    peak_month = stats["peak_month"]
    valley_month = stats["valley_month"]
    axes[0].axvline(peak_month, color="#800000", linestyle="--", linewidth=1.0, alpha=0.8)
    axes[0].axvline(valley_month, color="#006400", linestyle="--", linewidth=1.0, alpha=0.8)

    # (b) Month-hour heatmap to jointly show seasonal and intraday effects.
    month_hour = (
        df.groupby(["month", "hour"])["load"].mean().unstack("hour").reindex(index=range(1, 13), columns=range(24))
    )
    im = axes[1].imshow(month_hour.values, aspect="auto", origin="lower", cmap="YlOrRd")
    axes[1].set_title("(b) Month-Hour Mean Load Heatmap", fontweight="bold")
    axes[1].set_xlabel("Hour of Day")
    axes[1].set_ylabel("Month")
    axes[1].set_xticks(np.arange(0, 24, 4))
    axes[1].set_yticks(np.arange(12))
    axes[1].set_yticklabels(MONTH_LABELS)
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.022, pad=0.01)
    cbar.set_label("Mean Load (MW)")

    for ax in axes:
        _beautify_axis(ax)

    out = outdir / "pjme_seasonality.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def save_paper_figure(s: pd.Series, outdir: Path, width: float, height: float, dpi: int, stats: dict) -> None:
    daily = s.resample("D").mean()
    hour_stats = s.groupby(s.index.hour).agg(["mean", lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)])
    hour_stats.columns = ["mean", "q25", "q75"]
    month_mean = stats["month_mean"]

    max_lag = min(24 * 10, len(s) - 2)
    lags = np.arange(1, max_lag + 1)
    acf_vals = np.array([s.autocorr(lag=int(lag)) for lag in lags], dtype=float)

    fig = plt.figure(figsize=(width, height), dpi=dpi, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    ax1.plot(daily.index, daily.values, color="#2E4164", linewidth=0.8, alpha=0.8)
    ax1.plot(daily.index, daily.rolling(30, min_periods=1).mean(), color="#800000", linewidth=1.5)
    ax1.set_title("(a) Long-Term Trend", fontweight="bold")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Load (MW)")

    hrs = np.arange(24)
    ax2.plot(hrs, hour_stats["mean"], color="#2E4164", linewidth=1.8)
    ax2.fill_between(hrs, hour_stats["q25"], hour_stats["q75"], color="#5E7994", alpha=0.3)
    peak_h = stats["peak_hour"]
    valley_h = stats["valley_hour"]
    peak_y = hour_stats.loc[peak_h, "mean"]
    valley_y = hour_stats.loc[valley_h, "mean"]
    ax2.scatter([peak_h, valley_h], [peak_y, valley_y], color=["#800000", "#006400"], s=40, zorder=5, marker="o", edgecolors="white", linewidth=0.5)
    ax2.set_title("(b) Intraday Pattern", fontweight="bold")
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Load (MW)")
    ax2.set_xticks(np.arange(0, 24, 4))

    ax3.plot(lags, acf_vals, color="#2E4164", linewidth=1.2)
    ax3.axhline(0.0, color="#444444", linewidth=0.8)
    for special_lag, color in [(24, "#800000"), (168, "#006400")]:
        if special_lag <= max_lag:
            val = s.autocorr(lag=special_lag)
            ax3.axvline(special_lag, color=color, linestyle="--", linewidth=1.0)
            ax3.annotate(f"{special_lag}", xy=(special_lag, 0.05), fontsize=10, color=color, ha="center")
    ax3.set_title("(c) Autocorrelation", fontweight="bold")
    ax3.set_xlabel("Lag (hours)")
    ax3.set_ylabel("ACF")
    ax3.set_xlim(1, max_lag)
    ax3.set_ylim(-0.2, 1.02)

    months = np.arange(1, 13)
    ax4.plot(months, month_mean.values, color="#006400", marker="o", markersize=6, linewidth=2, markerfacecolor="#2E4164", markeredgewidth=0.5, markeredgecolor="black")
    ax4.scatter(
        [stats["peak_month"], stats["valley_month"]],
        [month_mean.loc[stats["peak_month"]], month_mean.loc[stats["valley_month"]]],
        color=["#800000", "#006400"],
        s=40,
        zorder=5,
        marker="o",
        edgecolors="white",
        linewidth=0.5,
    )
    ax4.set_title("(d) Yearly Seasonality", fontweight="bold")
    ax4.set_xlabel("Month")
    ax4.set_ylabel("Mean Load (MW)")
    ax4.set_xticks(months)
    ax4.set_xticklabels(MONTH_LABELS)

    fig.suptitle("PJME Load Characteristics", fontsize=16, fontweight="bold")

    for ax in [ax1, ax2, ax3, ax4]:
        _beautify_axis(ax)

    out = outdir / "pjme_paper_figure.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    args = parse_args()
    apply_style()

    args.outdir.mkdir(parents=True, exist_ok=True)

    s = load_series(args)
    stats = get_summary_stats(s)
    print_summary(stats)

    save_overview(s, args.outdir, args.width, args.height, args.dpi, stats)
    save_seasonality_figure(s, args.outdir, args.width, args.height, args.dpi, stats)
    save_acf_plot(s, args.outdir, args.width, args.height, args.dpi, args.max_lag)
    save_year_heatmap(s, args.outdir, args.width, args.height, args.dpi, args.heatmap_year)
    save_paper_figure(s, args.outdir, args.width, args.height, args.dpi, stats)


if __name__ == "__main__":
    main()
