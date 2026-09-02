from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
CONTROLLED_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "MPLBACKEND": "Agg",
    "PYTHONDONTWRITEBYTECODE": "1",
}


def run(command: list[str], environment: dict[str, str]) -> None:
    print(f"[RUN] {' '.join(command)}", flush=True)
    subprocess.run(
        command, cwd=PROJECT_ROOT, env=environment, check=True
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Robotica deterministic experiment chain."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    stage3_5_dir = output_dir / "stage3_5"
    stage3_3_dir = output_dir / "stage3_3"
    stage3_5_dir.mkdir(parents=True, exist_ok=True)
    stage3_3_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(CONTROLLED_ENVIRONMENT)

    run([
        sys.executable,
        str(SCRIPT_DIR / "run_final_experiments.py"),
        "--output-dir", str(stage3_5_dir),
    ], environment)
    for script in (
        "reproduce_stage3_3_p0_exact_alpha.py",
        "reproduce_stage3_3_p1_endpoints.py",
        "reproduce_stage3_3_p1_workload.py",
        "reproduce_stage3_3_p2_3_disagreement.py",
    ):
        run([
            sys.executable,
            str(SCRIPT_DIR / script),
            "--output-dir", str(stage3_3_dir),
        ], environment)
    print("ROBOTICA EXPERIMENT CHAIN: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

