"""Reproduce Fig. 3 from the frozen minimax-plateau sweep CSV."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from plot_utils import data_dir, default_output, save


def main(out: Path | None = None):
    df = pd.read_csv(data_dir() / "stage3_3_p1_minimax_plateau_sweep.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    for family, marker, label in [
        ("hub_m8_seed0", "o", "hub_m8_seed0"),
        ("intel_scLTL", "s", "intel_scLTL"),
    ]:
        sub = df[df.family == family].sort_values("alpha")
        ax.plot(sub.alpha, sub.dynamic_value, marker=marker, linewidth=1.4,
                markersize=4.5, label=label)
        ax.axhline(sub.minimax_value.iloc[0], linestyle="--", linewidth=0.9)
    ax.set_xlabel(r"Risk level $\alpha$")
    ax.set_ylabel("Optimal dynamic regret")
    ax.set_title("Instance-specific approach to the minimax-regret plateau")
    ax.legend(loc="lower right")

    fig.tight_layout()
    return save(fig, out or default_output("fig_stage3_3_p1_minimax_plateau.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(); print(main(args.out))
