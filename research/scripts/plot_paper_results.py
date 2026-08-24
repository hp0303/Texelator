#!/usr/bin/env python3
"""Regenerate compact paper figures from the frozen public CSV files."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import scienceplots  # noqa: F401

    plt.style.use(["science", "no-latex"])
except ImportError:
    plt.style.use("default")


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "paper"
OUTPUT = ROOT / "paper_figures"
OUTPUT.mkdir(exist_ok=True)

GRAY = "#8F8F8F"
LIGHT_GRAY = "#C4C4C4"
BLUE = "#56B4E9"
ORANGE = "#D55E00"


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.2,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "savefig.dpi": 300,
    })


def finish(fig: plt.Figure, name: str) -> None:
    for ax in fig.axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.65)
        ax.set_axisbelow(True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"{name}.{suffix}", bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def kernel_progression() -> None:
    rows = list(csv.DictReader((DATA / "kernel_progression.csv").open()))
    values = np.array([float(row["physical_weight_throughput_gb_s"]) for row in rows])
    labels = ["Marlin\nINT4", "BC4\ntex2D()", "BC4\ngather", "BC4\ngather + ILP"]
    fig, ax = plt.subplots(figsize=(3.42, 2.15))
    bars = ax.bar(range(4), values, width=0.68,
                  color=[GRAY, "#0072B2", BLUE, ORANGE],
                  edgecolor="#303030", linewidth=0.55)
    ax.set_ylabel("Physical weight throughput (GB/s)")
    ax.set_xticks(range(4), labels)
    ax.set_ylim(0, 215)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 5, f"{value:.1f}",
                ha="center", va="bottom", fontsize=7.2)
    finish(fig, "kernel_progression")


def context_512() -> None:
    rows = list(csv.DictReader((DATA / "rtx4080_context512.csv").open()))
    models = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B",
              "deepseek-ai/deepseek-coder-1.3b-base"]
    model_labels = ["Qwen\n0.5B", "Qwen\n1.5B", "Qwen\n3B", "DeepSeek-Coder\n1.3B"]
    variants = ["fp16", "gptq", "awq", "texelator"]
    names = ["FP16", "GPTQ", "AWQ", "Texelator"]
    lookup = {(row["model"], row["variant"]): row for row in rows}
    x = np.arange(len(models))
    width = 0.19
    fig, ax = plt.subplots(figsize=(7.1, 2.35))
    for index, (variant, name, color) in enumerate(zip(
            variants, names, [LIGHT_GRAY, GRAY, BLUE, ORANGE])):
        values = np.array([float(lookup[(model, variant)]["tokens_s_median"]) for model in models])
        low = np.array([float(lookup[(model, variant)]["tokens_s_min"]) for model in models])
        high = np.array([float(lookup[(model, variant)]["tokens_s_max"]) for model in models])
        ax.bar(x + (index - 1.5) * width, values, width, label=name, color=color,
               edgecolor="#303030", linewidth=0.45,
               yerr=np.vstack((values - low, high - values)), capsize=1.8,
               error_kw={"elinewidth": 0.7, "capthick": 0.7})
    ax.set_ylabel("Decode throughput (tokens/s)")
    ax.set_xticks(x, model_labels)
    ax.set_ylim(0, 240)
    ax.legend(ncol=4, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.01))
    finish(fig, "rtx4080_context512")


def context_speedup() -> None:
    rows = list(csv.DictReader((DATA / "rtx4080_context_speedup.csv").open()))
    fig, ax = plt.subplots(figsize=(5.2, 2.45))
    colors = ["#0072B2", "#009E73", ORANGE, GRAY]
    for color, model in zip(colors, dict.fromkeys(row["model"] for row in rows)):
        selected = [row for row in rows if row["model"] == model]
        ax.plot([int(row["context"]) for row in selected],
                [float(row["texelator_over_gptq"]) for row in selected],
                marker="o", markersize=3.3, label=model, color=color)
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=0.9)
    ax.set_xscale("log", base=2)
    ax.set_xticks([128, 512, 2048, 4096, 8192], ["128", "512", "2048", "4096", "8192"])
    ax.set_xlabel("Context length")
    ax.set_ylabel("Speedup over GPTQ-Marlin")
    ax.legend(frameon=False, ncol=2)
    finish(fig, "rtx4080_context_speedup")


if __name__ == "__main__":
    style()
    kernel_progression()
    context_512()
    context_speedup()
    print(f"Figures written to {OUTPUT}")
