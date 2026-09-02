from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reproduced_results"
CONTROLLED_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "MPLBACKEND": "Agg",
    "PYTHONDONTWRITEBYTECODE": "1",
}
EXPERIMENT_ORDER = (
    "horizon",
    "prior",
    "baselines",
    "mismatch",
    "timing",
    "intel-spec",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the final Stage 3.5 experiment and result-generation chain."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=EXPERIMENT_ORDER,
        default=EXPERIMENT_ORDER,
        help="Subset to run in canonical dependency order.",
    )
    return parser.parse_args()


def command_map(output_dir: Path) -> dict[str, list[str]]:
    python = sys.executable
    return {
        "horizon": [
            python,
            str(SCRIPT_DIR / "stage3_5_horizon_sensitivity.py"),
            "--output-dir",
            str(output_dir),
        ],
        "prior": [
            python,
            str(SCRIPT_DIR / "stage3_5_prior_robustness.py"),
            "--output-dir",
            str(output_dir),
        ],
        "baselines": [
            python,
            str(SCRIPT_DIR / "stage3_5_common_objective_baselines.py"),
            "--output-dir",
            str(output_dir),
        ],
        "mismatch": [
            python,
            str(SCRIPT_DIR / "stage3_5_static_nested_mismatch_sweep.py"),
            "--output-dir",
            str(output_dir),
        ],
        "timing": [
            python,
            str(SCRIPT_DIR / "stage3_5_controlled_timing_statistics.py"),
            "--output-csv",
            str(output_dir / "stage3_5_controlled_timing_statistics.csv"),
            "--output-figure",
            str(output_dir / "fig_stage3_5_controlled_timing_statistics.png"),
        ],
        "intel-spec": [
            python,
            str(SCRIPT_DIR / "stage3_5_export_intel_spec.py"),
            "--json-output",
            str(output_dir / "stage3_5_intel_spec.json"),
            "--tex-output",
            str(output_dir / "stage3_5_intel_spec_table.tex"),
        ],
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(CONTROLLED_ENVIRONMENT)
    commands = command_map(output_dir)
    selected = set(args.experiments)

    for name in EXPERIMENT_ORDER:
        if name not in selected:
            continue
        command = commands[name]
        print(f"[RUN:{name}] {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)

    print(f"FINAL EXPERIMENT CHAIN: PASS ({output_dir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
