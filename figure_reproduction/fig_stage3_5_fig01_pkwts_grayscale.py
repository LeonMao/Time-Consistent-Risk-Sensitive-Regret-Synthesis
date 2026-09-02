"""Generate the Stage 3.5 grayscale-safe variant of manuscript Fig. 1."""
from __future__ import annotations

import argparse
from pathlib import Path

from fig01_pkwts_time_consistency import main as draw_figure
from plot_utils import default_output


FILENAME = "fig_stage3_5_pkwts_time_consistency_grayscale.png"


def main(out: Path | None = None):
    return draw_figure(out or default_output(FILENAME))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    print(main(args.out))
