"""Reproduce Fig. 6(a): deterministic distributed layered benchmark schematic."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from plot_utils import default_output, save


def main(out: Path | None = None):
    layers = 5
    width = 2
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    y = {0: 2.2, 1: 1.0}
    x = {l: 1.0 + 1.15*(l-1) for l in range(1, layers+1)}
    # Start/goal.
    ax.add_patch(Rectangle((0.45, 1.35), 0.12, 0.12, fill=False, linewidth=1.0))
    ax.add_patch(Rectangle((6.45, 1.35), 0.12, 0.12, fill=False, linewidth=1.0))
    ax.text(0.51, 1.56, "s", ha="center", fontsize=9)
    ax.text(6.51, 1.56, "g", ha="center", fontsize=9)
    # Layer nodes.
    for l in range(1, layers+1):
        for j in range(width):
            ax.plot(x[l], y[j], marker="o", markersize=5,
                    color="black", markerfacecolor="white",
                    markeredgecolor="black", markeredgewidth=1.0)
            ax.text(x[l], y[j]+0.18, f"{l}_{j}", ha="center", fontsize=8.5)
    # Start/goal forward connections.
    ax.plot([0.57, x[1]], [1.41, y[0]], color="black", linewidth=1.0)
    ax.plot([0.57, x[1]], [1.41, y[1]], color="black", linewidth=1.0)
    ax.plot([x[5], 6.45], [y[0], 1.41], color="black", linewidth=1.0)
    ax.plot([x[5], 6.45], [y[1], 1.41], color="black", linewidth=1.0)
    # Solid forward edges: both same-index successors.
    for l in range(1, layers):
        for j in range(width):
            ax.plot([x[l], x[l+1]], [y[j], y[j]], color="black",
                    linestyle="-", linewidth=1.0)
    # Representative open-only shortcuts use dashed lines and diamond midpoints.
    for l in range(1, layers-1):
        for source_y, target_y in ((y[0], y[1]), (y[1], y[0])):
            ax.plot([x[l], x[l+2]], [source_y, target_y], color="0.35",
                    linestyle="--", linewidth=0.9)
            ax.plot((x[l] + x[l+2]) / 2, (source_y + target_y) / 2,
                    marker="D", markerfacecolor="white",
                    markeredgecolor="black", markersize=3.5)
    ax.text(3.45, 0.30,
            "solid + circles: closed-mode forward edges    dashed + diamonds: open-only shortcuts",
            ha="center", fontsize=8.5)
    ax.set_xlim(0.2, 6.85); ax.set_ylim(0.05, 2.75)
    ax.axis("off")
    return save(fig, out or default_output("fig_stage3_2_layered_benchmark.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(); print(main(args.out))
