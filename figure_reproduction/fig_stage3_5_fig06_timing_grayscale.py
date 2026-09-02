"""Generate the grayscale-safe paired-timing panel used in manuscript Fig. 6(b)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_utils import data_dir, default_output, save


FILENAME = "fig_stage3_5_controlled_timing_statistics_grayscale.png"
SUMMARY_PATH = data_dir() / "stage3_5_controlled_timing_statistics.csv"
RAW_PATH = data_dir() / "stage3_2_controlled_benchmark_raw.csv"


def seed_ratios(raw: pd.DataFrame, family: str, m_value: int) -> list[float]:
    subset = raw.loc[(raw["family"] == family) & (raw["m"] == m_value)]
    medians = subset.groupby(["seed", "method"], sort=True)["solve_s"].median().unstack()
    return (medians["explicit"] / medians["factored"]).tolist()


def main(out: Path | None = None):
    summary = pd.read_csv(SUMMARY_PATH)
    raw = pd.read_csv(RAW_PATH)
    aggregate = summary.loc[summary["aggregate_role"] == "paired_five_seed"]
    stress = summary.loc[summary["aggregate_role"] == "single_seed_stress_excluded"].iloc[0]
    styles = {
        "hub": {
            "color": "black",
            "linestyle": "-",
            "marker": "o",
            "seed_marker": "+",
            "offset": -0.12,
        },
        "layered": {
            "color": "0.35",
            "linestyle": "--",
            "marker": "s",
            "seed_marker": "x",
            "offset": 0.12,
        },
    }
    with plt.rc_context(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    ):
        figure, axis = plt.subplots(figsize=(6.0, 4.0))
        for family in ("hub", "layered"):
            rows = aggregate.loc[aggregate["family"] == family].sort_values("m")
            style = styles[family]
            x_values = rows["m"].astype(int).to_numpy() + style["offset"]
            medians = rows["paired_speedup_median_x"].astype(float).to_numpy()
            axis.plot(
                x_values,
                medians,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markerfacecolor="white",
                markeredgecolor="black",
                linewidth=1.4,
                markersize=5.5,
                label=f"{family.capitalize()} aggregate (n=5)",
                zorder=4,
            )
            for x_value, (_, row) in zip(x_values, rows.iterrows()):
                point = float(row["paired_speedup_median_x"])
                low = float(row["paired_speedup_bootstrap_ci95_low_x"])
                high = float(row["paired_speedup_bootstrap_ci95_high_x"])
                axis.errorbar(
                    x_value,
                    point,
                    yerr=[[point - low], [high - point]],
                    color=style["color"],
                    capsize=3,
                    linewidth=1.1,
                    zorder=3,
                )
                ratios = seed_ratios(raw, family, int(row["m"]))
                jitters = (-0.06, -0.03, 0.0, 0.03, 0.06)
                axis.scatter(
                    [x_value + jitter for jitter in jitters],
                    ratios,
                    color=style["color"],
                    marker=style["seed_marker"],
                    s=20,
                    linewidths=0.8,
                    zorder=2,
                )

        stress_x = int(stress["m"]) + styles["layered"]["offset"]
        stress_ratio = float(stress["single_seed_observed_ratio_x"])
        axis.scatter(
            [stress_x],
            [stress_ratio],
            marker="^",
            facecolors="white",
            edgecolors="black",
            linewidths=1.3,
            s=54,
            label="Layered stress point (n=1; no CI)",
            zorder=5,
        )
        axis.annotate(
            "stress only",
            (stress_x, stress_ratio),
            xytext=(-10, 9),
            textcoords="offset points",
            ha="right",
            color="black",
            fontsize=7.5,
        )
        axis.axhline(1.0, color="0.4", linewidth=0.8, linestyle=":")
        axis.set_yscale("log")
        axis.set_xlim(7.45, 12.65)
        axis.set_ylim(1.0, 55.0)
        axis.set_xticks((8, 10, 12))
        axis.set_xlabel("Unknown topology variables $m$")
        axis.set_ylabel("Paired explicit / factored solve-time ratio")
        axis.legend(loc="upper left", frameon=False)
        axis.grid(axis="y", which="major", color="0.85", linewidth=0.6)
        axis.grid(axis="y", which="minor", visible=False)
        figure.tight_layout(pad=0.5)
        return save(figure, out or default_output(FILENAME), dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    print(main(args.out))
