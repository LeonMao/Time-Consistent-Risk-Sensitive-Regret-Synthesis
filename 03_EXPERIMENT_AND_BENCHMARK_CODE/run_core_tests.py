from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = PROJECT_ROOT / "02_CORE_CODE"
VERIFY_SCRIPT = Path(__file__).with_name("verify_research_environment.py")
CONTROLLED_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "MPLBACKEND": "Agg",
    "PYTHONDONTWRITEBYTECODE": "1",
}


def run(command: list[str], cwd: Path, environment: dict[str, str]) -> None:
    print(f"[RUN] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def main() -> int:
    environment = os.environ.copy()
    environment.update(CONTROLLED_ENVIRONMENT)

    run(
        [sys.executable, str(VERIFY_SCRIPT), "--require-controls"],
        PROJECT_ROOT,
        environment,
    )

    tests = sorted(CORE_DIR.glob("test_*.py"))
    if not tests:
        raise RuntimeError("No core test scripts were found.")

    for test in tests:
        run([sys.executable, test.name], CORE_DIR, environment)

    print(f"CORE TEST SCRIPTS: ALL {len(tests)} PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
