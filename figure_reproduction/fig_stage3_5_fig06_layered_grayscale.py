"""Generate the Stage 3.5 grayscale-safe variant of manuscript Fig. 6(a)."""
from __future__ import annotations

import argparse
from pathlib import Path

from fig06_layered_benchmark import main as draw_figure
from plot_utils import default_output


FILENAME = "fig_stage3_5_layered_benchmark_grayscale.png"


def main(out: Path | None = None):
    return draw_figure(out or default_output(FILENAME))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    print(main(args.out))
