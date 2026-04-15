#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize PJME hourly load data with configurable range and figure size.

Examples:
  python visualize_pjme.py --start "2005-01-01" --end "2005-03-01"
  python visualize_pjme.py --iloc-start 1000 --iloc-end 3000 --width 14 --height 5
  python visualize_pjme.py --start "2010-01-01" --end "2010-12-31" --output pjme_2010.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize PJME_hourly.csv with academic-style plotting."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("PJME_hourly.csv"),
        help="Path to CSV file (default: PJME_hourly.csv)",
    )
    parser.add_argument(
        "--datetime-col",
        type=str,
        default="Datetime",
        help="Datetime column name (default: Datetime)",
    )
    parser.add_argument(
        "--value-col",
        type=str,
        default="PJME_MW",
        help="Value column name (default: PJME_MW)",
    )

    # Range by datetime
    parser.add_argument("--start", type=str, default=None, help="Start datetime, e.g. 2005-01-01")
    parser.add_argument("--end", type=str, default=None, help="End datetime, e.g. 2005-03-01")

    # Range by iloc index
    parser.add_argument("--iloc-start", type=int, default=None, help="Start row index (inclusive)")
    parser.add_argument("--iloc-end", type=int, default=None, help="End row index (exclusive)")

    # Figure controls
    parser.add_argument("--width", type=float, default=12.0, help="Figure width in inches")
    parser.add_argument("--height", type=float, default=4.8, help="Figure height in inches")
    parser.add_argument("--dpi", type=int, default=140, help="Figure DPI")
    parser.add_argument(
        "--title",
        type=str,
        default="PJME Hourly Power Load",
        help="Plot title",
    )

    parser.add_argument(
        "--rolling",
        type=int,
        default=24,
        help="Rolling mean window in hours (0 to disable, default: 24)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. If omitted, display interactively.",
    )
    return parser.parse_args()


def apply_academic_style() -> None:
    # A clean, publication-like style.
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "STSong"],
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.0,
            "grid.color": "#C7C7C7",
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "lines.linewidth": 1.6,
        }
    )


def load_and_filter_data(args: argparse.Namespace) -> pd.DataFrame:
    if not args.csv.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv}")

    df = pd.read_csv(args.csv)

    if args.datetime_col not in df.columns or args.value_col not in df.columns:
        raise ValueError(
            f"Columns not found. Available columns: {list(df.columns)}; "
            f"expected datetime='{args.datetime_col}', value='{args.value_col}'"
        )

    df[args.datetime_col] = pd.to_datetime(df[args.datetime_col], errors="coerce")
    df = df.dropna(subset=[args.datetime_col, args.value_col]).copy()
    df = df.sort_values(args.datetime_col)

    # Datetime range filter
    if args.start is not None:
        start_dt = pd.to_datetime(args.start)
        df = df[df[args.datetime_col] >= start_dt]
    if args.end is not None:
        end_dt = pd.to_datetime(args.end)
        df = df[df[args.datetime_col] <= end_dt]

    # iloc range filter
    if args.iloc_start is not None or args.iloc_end is not None:
        df = df.iloc[args.iloc_start : args.iloc_end]

    if df.empty:
        raise ValueError("No data left after filtering. Please check your range arguments.")

    df = df.set_index(args.datetime_col)
    return df


def plot_data(df: pd.DataFrame, args: argparse.Namespace) -> None:
    apply_academic_style()

    fig, ax = plt.subplots(figsize=(args.width, args.height), dpi=args.dpi)

    series = df[args.value_col]
    ax.plot(series.index, series.values, color="#1f4e79", alpha=0.9, label="Hourly load")

    if args.rolling and args.rolling > 1:
        rolling_series = series.rolling(args.rolling, min_periods=1).mean()
        ax.plot(
            rolling_series.index,
            rolling_series.values,
            color="#b22222",
            linewidth=2.0,
            alpha=0.9,
            label=f"{args.rolling}-hour rolling mean",
        )

    ax.set_title(args.title)
    ax.set_xlabel("Datetime")
    ax.set_ylabel(args.value_col)
    ax.legend(frameon=True, framealpha=0.92)

    # Keep a clean publication-like frame.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.autofmt_xdate(rotation=20)
    plt.tight_layout()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, bbox_inches="tight")
        print(f"Saved plot to: {args.output}")
    else:
        plt.show()


def main() -> None:
    args = parse_args()
    df = load_and_filter_data(args)
    plot_data(df, args)


if __name__ == "__main__":
    main()
