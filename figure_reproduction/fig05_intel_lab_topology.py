"""Reproduce Fig. 5: schematic Intel Research Lab topological abstraction.

The benchmark geometry is taken directly from
03_EXPERIMENT_AND_BENCHMARK_CODE/stage3_3_intel_lab_benchmark.py.
The drawing layout is deterministic and intentionally schematic, as stated in
 the manuscript caption; it is not a reproduction of the original occupancy image.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from plot_utils import default_output, save


def main(out: Path | None = None):
    coords = {
        "s": (2.8, 0.5), "bl": (1.2, 0.8), "lm": (0.8, 2.4), "lt": (1.0, 4.2),
        "tm": (3.0, 4.7), "tr": (5.0, 4.4), "rm": (5.6, 2.7), "br": (5.2, 0.9),
        "ext": (0.15, 2.6), "fire": (6.35, 2.7),
        "d0": (1.8, 1.55), "d1": (1.85, 3.7), "d2": (4.15, 3.75),
        "d3": (4.25, 1.55), "d4": (3.25, 2.7),
    }
    safe_edges = [
        ("s", "bl"), ("bl", "lm"), ("lm", "lt"), ("lt", "tm"),
        ("tm", "tr"), ("tr", "rm"), ("rm", "br"), ("br", "s"),
        ("lm", "ext"), ("rm", "fire"),
    ]
    shortcuts = {"d0": ("s", "lm"), "d1": ("lm", "tm"), "d2": ("tm", "rm"),
                 "d3": ("s", "rm"), "d4": ("lt", "rm")}

    fig, ax = plt.subplots(figsize=(8.3, 5.4))
    # Perimeter/known connections.
    for u, v in safe_edges:
        ax.plot([coords[u][0], coords[v][0]], [coords[u][1], coords[v][1]],
                color="black", linestyle="-", linewidth=1.4)
    # Passage approach and open-only traversal use distinct grayscale-safe styles.
    for d, (u, v) in shortcuts.items():
        ax.plot([coords[u][0], coords[d][0]], [coords[u][1], coords[d][1]],
                color="0.45", linestyle=":", linewidth=1.1)
        ax.plot([coords[d][0], coords[v][0]], [coords[d][1], coords[v][1]],
                color="black", linestyle="--", linewidth=1.3)
    for name, (x, y) in coords.items():
        if name.startswith("d"):
            ax.plot(x, y, marker="D", markersize=7, markerfacecolor="white",
                    markeredgecolor="black", markeredgewidth=1.2)
            ax.text(x, y+0.18, name, ha="center", va="bottom", fontsize=9)
        elif name == "ext":
            ax.add_patch(Rectangle((x-0.11, y-0.11), 0.22, 0.22,
                                   facecolor="white", edgecolor="black",
                                   hatch="///", linewidth=1.1))
            ax.text(x, y-0.17, "Extinguisher", ha="center", va="top", fontsize=9)
        elif name == "fire":
            ax.add_patch(Rectangle((x-0.11, y-0.11), 0.22, 0.22,
                                   facecolor="0.75", edgecolor="black",
                                   hatch="xx", linewidth=1.1))
            ax.text(x+0.15, y, "Fire", ha="left", va="center", fontsize=9)
        else:
            ax.plot(x, y, marker="o", markersize=5, markerfacecolor="white",
                    markeredgecolor="black", markeredgewidth=1.0)
            ax.text(x, y+0.17, name, ha="center", va="bottom", fontsize=9)
    legend_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=1.4,
               label="known corridor"),
        Line2D([0], [0], color="0.45", linestyle=":", linewidth=1.1,
               label="probe approach"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.3,
               marker="D", markerfacecolor="white", markersize=5,
               label="open-only shortcut"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", ncol=3,
              bbox_to_anchor=(0.5, 0.01), frameon=False, fontsize=8)
    ax.set_title("Intel Research Lab map-derived topological benchmark (schematic abstraction)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.35, 6.9); ax.set_ylim(-0.45, 5.15)
    ax.axis("off")
    return save(fig, out or default_output("fig_stage3_3_intel_lab_topology.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(); print(main(args.out))
