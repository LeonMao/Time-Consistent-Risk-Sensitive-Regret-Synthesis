from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
CONTROLLED_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "MPLBACKEND": "Agg",
    "PYTHONDONTWRITEBYTECODE": "1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Robotica reproducibility and claim-verification gates."
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--quick", action="store_true", help="Run core tests and frozen claim checks (default).")
    modes.add_argument("--full", action="store_true", help="Recompute all method/experiment CSVs and compare them with the references.")
    modes.add_argument("--figures", action="store_true", help="Regenerate and verify the seven data-driven figures.")
    modes.add_argument("--claims", action="store_true", help="Verify the paper's headline claims from the frozen evidence.")
    modes.add_argument("--all", action="store_true", help="Run core tests, full recomputation, claims, and result-figure reproduction.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts",
        help="Fresh directory for generated results and figures.",
    )
    parser.add_argument(
        "--timeout-minutes",
        type=float,
        default=240.0,
        help="Hard timeout for each top-level phase.",
    )
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="Allow a non-empty output directory and overwrite named outputs.",
    )
    return parser.parse_args()


def run(label: str, command: list[str], environment: dict[str, str], timeout: float) -> None:
    print(f"[RUN:{label}] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True, timeout=timeout)


def prepare_output_root(path: Path, allow_existing: bool) -> Path:
    resolved = path.resolve()
    if resolved.exists() and any(resolved.iterdir()) and not allow_existing:
        raise RuntimeError(
            f"Output directory is not empty: {resolved}. Choose a fresh path or pass --allow-existing."
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def main() -> int:
    args = parse_args()
    no_mode = not any((args.quick, args.full, args.figures, args.claims, args.all))
    run_core = no_mode or args.quick or args.full or args.all
    run_full = args.full or args.all
    run_figures = args.figures or args.all
    run_claims = no_mode or args.quick or args.full or args.claims or args.all
    timeout = args.timeout_minutes * 60.0

    environment = os.environ.copy()
    environment.update(CONTROLLED_ENVIRONMENT)
    output_root = None
    if run_full or run_figures:
        output_root = prepare_output_root(args.output_root, args.allow_existing)

    if run_core:
        run(
            "core",
            [sys.executable, "03_EXPERIMENT_AND_BENCHMARK_CODE/run_core_tests.py"],
            environment,
            timeout,
        )

    if run_full:
        results_dir = output_root / "results"
        run(
            "robotica-experiments",
            [
                sys.executable,
                "03_EXPERIMENT_AND_BENCHMARK_CODE/run_robotica_experiments.py",
                "--output-dir",
                str(results_dir),
            ],
            environment,
            timeout,
        )
        run(
            "stage3.5-result-verification",
            [sys.executable, "verify_reproduced_results.py", str(results_dir / "stage3_5")],
            environment,
            timeout,
        )
        run(
            "stage3.3-result-verification",
            [sys.executable, "verify_robotica_recomputed.py", str(results_dir / "stage3_3")],
            environment,
            timeout,
        )

    if run_claims:
        run(
            "paper-claim-verification",
            [sys.executable, "verify_robotica_claims.py"],
            environment,
            timeout,
        )

    if run_figures:
        figures_dir = output_root / "figures"
        run(
            "result-figure-generation",
            [
                sys.executable,
                "figure_reproduction/reproduce_all_figures.py",
                "--output-dir",
                str(figures_dir),
            ],
            environment,
            timeout,
        )
        run(
            "result-figure-verification",
            [
                sys.executable,
                "figure_reproduction/verify_final_figures.py",
                "--generated-dir",
                str(figures_dir),
            ],
            environment,
            timeout,
        )

    print("ROBOTICA REPRODUCIBILITY GATES: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
