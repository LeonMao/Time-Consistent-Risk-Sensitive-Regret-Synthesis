"""Reproduce Fig. 1: sequential-revelation PK-WTS time-consistency schematic.

Source model:
    03_EXPERIMENT_AND_BENCHMARK_CODE/stage3_3_p2_sequential_pkwts.py
Frozen numerical results:
    06_DATA_AND_RESULTS/stage3_3_p2_static_dynamic_pkwts_results.csv
    06_DATA_AND_RESULTS/stage3_3_p2_time_consistency_summary.csv

This is a controlled schematic rather than a solver-generated graph layout.
The script reconstructs the exact four-world example and its reported values.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch

from plot_utils import data_dir, default_output, save


def draw_node(ax, xy, kind="square", label=None, size=0.07, hatch=None):
    x, y = xy
    if kind == "square":
        patch = Rectangle((x-size, y-size), 2*size, 2*size,
                          facecolor="white", edgecolor="black", hatch=hatch,
                          linewidth=1.0)
        ax.add_patch(patch)
    elif kind == "circle":
        patch = Circle((x, y), size, facecolor="white", edgecolor="black",
                       hatch=hatch, linewidth=1.0)
        ax.add_patch(patch)
    elif kind == "goal":
        patch = Circle((x, y), size, facecolor="white", edgecolor="black",
                       linewidth=1.0)
        ax.add_patch(patch)
        ax.plot([x], [y], marker="*", markersize=7, markerfacecolor="white",
                markeredgecolor="black")
    elif kind == "observe":
        ax.plot([x], [y], marker="D", markersize=5, markerfacecolor="white",
                markeredgecolor="black", markeredgewidth=1.0)
    if label:
        ax.text(x, y+0.16, label, ha="center", va="bottom", fontsize=9)


def main(out: Path | None = None):
    results = pd.read_csv(data_dir() / "stage3_3_p2_static_dynamic_pkwts_results.csv")
    summary = pd.read_csv(data_dir() / "stage3_3_p2_time_consistency_summary.csv")

    a = results.loc[results["candidate"] == 1].iloc[0]
    b = results.loc[results["candidate"] == 0].iloc[0]
    static = summary.iloc[0]
    nested = summary.iloc[1]

    # Fixed schematic coordinates. They are deliberately independent of graph-layout
    # packages so the figure is reproducible on every machine.
    p = {
        "start": (0.6, 2.1), "obs": (1.55, 2.1),
        "H": (2.55, 3.15), "L": (2.55, 1.05),
        "A": (3.85, 3.75), "B": (3.85, 2.65), "C": (3.85, 1.05),
        "gA": (5.25, 3.75), "gB": (5.25, 2.65), "gC": (5.25, 1.05),
    }

    fig, ax = plt.subplots(figsize=(10.0, 5.7))
    ax.set_xlim(0.2, 8.0)
    ax.set_ylim(0.35, 4.75)
    ax.axis("off")

    def edge(u, v, text=None, text_xy=None, linestyle="-"):
        ax.annotate("", xy=p[v], xytext=p[u],
                    arrowprops=dict(arrowstyle="->", linewidth=1.0,
                                    color="black", linestyle=linestyle,
                                    shrinkA=4, shrinkB=4))
        if text:
            tx, ty = text_xy if text_xy else ((p[u][0]+p[v][0])/2, (p[u][1]+p[v][1])/2)
            ax.text(tx, ty, text, fontsize=9, ha="center", va="center")

    draw_node(ax, p["start"], "square")
    draw_node(ax, p["obs"], "observe")
    draw_node(ax, p["H"], "circle")
    draw_node(ax, p["L"], "circle", hatch="///")
    draw_node(ax, p["A"], "square")
    draw_node(ax, p["B"], "square")
    draw_node(ax, p["C"], "square", hatch="///")
    draw_node(ax, p["gA"], "goal")
    draw_node(ax, p["gB"], "goal")
    draw_node(ax, p["gC"], "goal")

    edge("start", "obs", "")
    edge("obs", "H", "0.05", (2.0, 2.75))
    edge("obs", "L", "0.95", (2.0, 1.40), linestyle="-.")
    edge("H", "A", "route A", (3.15, 3.55))
    edge("H", "B", "route B", (3.15, 2.85))
    edge("L", "C", "route C", (3.15, 1.05), linestyle="-.")
    edge("A", "gA")
    edge("B", "gB")
    edge("C", "gC", linestyle="-.")

    ax.text(*p["start"], "", ha="center")
    ax.text(0.58, 2.35, "start", ha="center", fontsize=9)
    ax.text(1.55, 2.38, "observe $Y$", ha="center", fontsize=9)
    ax.text(2.55, 3.55, "rare $H$", ha="center", fontsize=9)
    ax.text(2.55, 0.70, "common $L$", ha="center", fontsize=9)
    ax.text(5.25, 4.00, "goal", ha="center", fontsize=9)
    ax.text(5.25, 2.90, "goal", ha="center", fontsize=9)
    ax.text(5.25, 1.30, "goal", ha="center", fontsize=9)

    ax.text(4.55, 3.95, r"$R=(20,3)$", fontsize=9)
    ax.text(4.55, 2.75, r"$R=(16,17)$", fontsize=9)
    ax.text(4.55, 0.85, r"$R=(6,9)$", fontsize=9)

    static_box = (
        "[S] Static CVaR$_{.75}$\n"
        "precommitment: choose A at H\n"
        "After observing H:\n"
        "CVaR$_{.75}$(A)=20\n"
        "CVaR$_{.75}$(B)=17\n"
        "replan to B"
    )
    nested_box = (
        "[N] Nested CVaR$_{.75}$:\n"
        "preselect B at H\n\n"
        "At H: B remains optimal\n"
        "mismatch probability = 0"
    )
    ax.add_patch(FancyBboxPatch((5.85, 2.35), 1.85, 1.65,
                                boxstyle="round,pad=0.04", facecolor="white",
                                edgecolor="black", linewidth=0.8,
                                linestyle="-"))
    ax.text(6.78, 3.18, static_box, ha="center", va="center", fontsize=7.8, linespacing=1.25)
    ax.add_patch(FancyBboxPatch((5.85, 0.55), 1.85, 1.45,
                                boxstyle="round,pad=0.04", facecolor="white",
                                edgecolor="black", linewidth=0.8,
                                linestyle="--"))
    ax.text(6.78, 1.28, nested_box, ha="center", va="center", fontsize=7.8, linespacing=1.25)

    ax.text(4.0, 0.15,
            rf"Static precommitment mismatch probability: {float(static.mismatch_probability):g}"
            rf"    Nested dynamic mismatch probability: {float(nested.mismatch_probability):g}",
            fontsize=8.5, ha="center")
    
    fig.tight_layout()
    return save(fig, out or default_output("fig_stage3_3_p2_static_dynamic_pkwts.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    print(main(args.out))
