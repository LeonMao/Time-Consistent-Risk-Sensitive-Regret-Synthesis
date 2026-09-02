"""Reproduce Fig. 2 from the frozen Stage 1.7 alpha summary CSV."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from plot_utils import data_dir, default_output, save


def main(out: Path | None = None):
    df = pd.read_csv(data_dir() / "stage1_7_summary_alpha.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(df.alpha, df.mean_regret_mean, yerr=df.mean_regret_std,
                marker="o", linewidth=1.4, capsize=3, label="Exact mean regret")
    ax.errorbar(df.alpha, df.cvar95_regret_mean, yerr=df.cvar95_regret_std,
                marker="s", linewidth=1.4, capsize=3, label=r"Exact static CVaR$_{.95}$ regret")
    ax.errorbar(df.alpha, df.dynamic_objective_mean, yerr=df.dynamic_objective_std,
                marker="^", linewidth=1.4, capsize=3, label="Nested planning objective")
    ax.set_xlabel(r"Planning risk level $\alpha$")
    ax.set_ylabel("Regret")
    ax.set_title("Exact finite-world risk evaluation (10 seeds, 256 worlds/seed)")
    ax.legend(loc="upper right")
    ax.grid(False)

    fig.tight_layout()
    return save(fig, out or default_output("fig_stage3_3_exact_alpha_tradeoff.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(); print(main(args.out))
