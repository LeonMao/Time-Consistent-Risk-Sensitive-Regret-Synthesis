"""Shared utilities for reproducing the seven figures used in the final T-RO manuscript.

The scripts intentionally use only matplotlib/pandas/numpy for figure generation.
No solver code is imported by plotting scripts unless a benchmark schematic needs
source geometry. All data-driven figures read the frozen CSV files in
06_DATA_AND_RESULTS so that published numerical values are not changed by
machine-dependent re-timing.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt


def project_root() -> Path:
    """Return the project root (parent of figure_reproduction)."""
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    return project_root() / "06_DATA_AND_RESULTS"


def figure_dir() -> Path:
    return project_root() / "01_FIGURES"


def generated_dir() -> Path:
    return Path(__file__).resolve().parent / "generated"

def default_output(name: str) -> Path:
    return generated_dir() / name


def save(fig, path: Path, dpi: int = 240) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
