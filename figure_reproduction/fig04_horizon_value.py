"""Reproduce Fig. 4 from the frozen Stage 1.7 horizon summary CSV."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from plot_utils import data_dir, default_output, save


def main(out: Path | None = None):
    df = pd.read_csv(data_dir() / "stage1_7_summary_horizon.csv").sort_values("H")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(df.H, df.dynamic_objective_mean, yerr=df.dynamic_objective_std,
                marker="o", linewidth=1.5, capsize=3)
    ax.set_xlabel(r"Robust completion horizon $H$")
    ax.set_ylabel("Dynamic regret objective")
    ax.set_title(r"Larger $H$ enlarges the proper policy class")
    ax.grid(False)

    fig.tight_layout()
    return save(fig, out or default_output("fig_stage1_7_horizon_value.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(); print(main(args.out))
