#!/usr/bin/env python3

"""MPI analysis post-process adapted to include detailed phase timings."""

import argparse
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd


def _parse_args():
    p = argparse.ArgumentParser(
        description="Post-processor of test_parallel.py with phase breakdown"
    )
    p.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path(__file__).parent / "data" / "mpi_analysis.csv",
    )
    p.add_argument(
        "-s",
        "--summary",
        type=Path,
        default=Path(__file__).parent / "data" / "mpi_analysis_summary.csv",
    )
    p.add_argument("-p", "--prefix", type=str, default="mpi")
    return p.parse_args()


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute data summary including total and phase timings."""
    df = df.astype(
        {
            "time_ns": float,
            "time_mesh_gen_ns": float,
            "time_spaces_def_ns": float,
            "time_bcs_def_ns": float,
            "time_baseflow_compute_ns": float,
            "time_assemble_ns": float,
            "peak_rss_MB": float,
            "peak_vms_MB": float,
        }
    )

    df["time_s"] = df["time_ns"] / 1e9
    df["mesh_gen_s"] = df["time_mesh_gen_ns"] / 1e9
    df["spaces_def_s"] = df["time_spaces_def_ns"] / 1e9
    df["bcs_def_s"] = df["time_bcs_def_ns"] / 1e9
    df["baseflow_compute_s"] = df["time_baseflow_compute_ns"] / 1e9
    df["assemble_s"] = df["time_assemble_ns"] / 1e9

    df["rss_GB"] = df["peak_rss_MB"] / 1024
    df["vms_GB"] = df["peak_vms_MB"] / 1024
    df["rss_per_rank_MB"] = df["peak_rss_MB"] / df["cores"]

    summary = (
        df.groupby("cores")
        .agg(
            runs=("run", "count"),
            time_mean=("time_s", "mean"),
            time_std=("time_s", "std"),
            mesh_gen_mean=("mesh_gen_s", "mean"),
            mesh_gen_std=("mesh_gen_s", "std"),
            spaces_def_mean=("spaces_def_s", "mean"),
            spaces_def_std=("spaces_def_s", "std"),
            bcs_def_mean=("bcs_def_s", "mean"),
            bcs_def_std=("bcs_def_s", "std"),
            baseflow_compute_mean=("baseflow_compute_s", "mean"),
            baseflow_compute_std=("baseflow_compute_s", "std"),
            assemble_mean=("assemble_s", "mean"),
            assemble_std=("assemble_s", "std"),
            rss_mean=("rss_GB", "mean"),
            rss_std=("rss_GB", "std"),
            vms_mean=("vms_GB", "mean"),
            vms_std=("vms_GB", "std"),
            rsspr_mean=("rss_per_rank_MB", "mean"),
        )
        .reset_index()
    )

    t1 = summary.loc[summary.cores == 1, "time_mean"].iloc[0]
    summary["speedup"] = t1 / summary["time_mean"]
    summary["efficiency"] = summary["speedup"] / summary["cores"]

    return summary


def print_summary(s: pd.DataFrame) -> None:
    """Print summary to console."""
    cols = [
        "cores",
        "runs",
        "time_mean",
        "time_std",
        "speedup",
        "efficiency",
        "mesh_gen_mean",
        "mesh_gen_std",
        "spaces_def_mean",
        "spaces_def_std",
        "bcs_def_mean",
        "bcs_def_std",
        "baseflow_compute_mean",
        "baseflow_compute_std",
        "assemble_mean",
        "assemble_std",
        "rss_mean",
        "rss_std",
        "rsspr_mean",
    ]
    print("\nMPI Benchmark Summary with Phase Breakdown:")
    print(s[cols].to_string(index=False, float_format="%.3f"))


def save_summary(s: pd.DataFrame, path: Path) -> None:
    """Save summary to CSV."""
    s.to_csv(path, index=False)
    print(f"Saved summary to {path}")


def _plot(
    x: list[float],
    y: dict[str, list[float]],
    ylabel: str,
    fname: str,
    ideal: Callable[[list[float]], list[float]] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    if ideal:
        ax.plot(x, ideal(x), "--", lw=1, color="0.3", label="Ideal")
    ax.errorbar(x, y["mean"], yerr=y.get("std"), marker="o", ls="-", color="k")
    ax.set(xlabel="MPI Ranks", ylabel=ylabel)
    ax.minorticks_on()
    ax.grid(which="major", ls="-", lw=0.5, color="0.7")
    ax.grid(which="minor", ls="--", lw=0.3, color="0.7")
    ax.tick_params(direction="in", which="both", labelsize=10)
    if ylim:
        ax.set_ylim(*ylim)
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot {fname}")


def main() -> None:
    """Script entry point."""
    args = _parse_args()
    df = pd.read_csv(args.input)
    summary = compute_summary(df)

    print_summary(summary)
    save_summary(summary, args.summary)

    cores = summary["cores"]
    _plot(
        cores,
        {"mean": summary["time_mean"], "std": summary["time_std"]},
        "Time (s)",
        f"{args.prefix}_time.png",
    )
    _plot(
        cores,
        {"mean": summary["speedup"]},
        "Performance (-)",
        f"{args.prefix}_speedup.png",
        # ideal=lambda x: x,
    )
    _plot(
        cores,
        {"mean": summary["efficiency"]},
        "Efficiency (-)",
        f"{args.prefix}_efficiency.png",
        ylim=(0, 1.05),
    )
    _plot(
        cores,
        {"mean": summary["rss_mean"], "std": summary["rss_std"]},
        "Peak RSS (GB)",
        f"{args.prefix}_rss.png",
    )
    _plot(
        cores,
        {"mean": summary["rsspr_mean"]},
        "RSS per rank (MB)",
        f"{args.prefix}_rsspr.png",
    )

    # Phase timings
    phases = [
        ("mesh_gen", "Mesh Generation (s)"),
        ("spaces_def", "Spaces Definition (s)"),
        ("bcs_def", "BCs Definition (s)"),
        ("baseflow_compute", "Baseflow Compute (s)"),
        ("assemble", "Assembly (s)"),
    ]
    for key, label in phases:
        _plot(
            cores,
            {"mean": summary[f"{key}_mean"], "std": summary[f"{key}_std"]},
            label,
            f"{args.prefix}_{key}.png",
        )


if __name__ == "__main__":
    main()
