from __future__ import annotations

import argparse
import importlib
from importlib import metadata
import json
import os
import platform
import sys


EXPECTED_PYTHON = "3.13.5"
EXPECTED_PACKAGES = {
    "numpy": ("numpy", "2.3.5"),
    "scipy": ("scipy", "1.17.0"),
    "psutil": ("psutil", "7.2.2"),
    "pandas": ("pandas", "3.0.1"),
    "matplotlib": ("matplotlib", "3.11.1"),
    "Pillow": ("PIL", "12.3.0"),
}
CONTROLLED_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "MPLBACKEND": "Agg",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-controls", action="store_true")
    args = parser.parse_args()

    errors = []
    actual_python = platform.python_version()
    if actual_python != EXPECTED_PYTHON:
        errors.append(f"Python {actual_python} != {EXPECTED_PYTHON}")

    package_versions = {}
    for distribution, (module_name, expected_version) in EXPECTED_PACKAGES.items():
        try:
            importlib.import_module(module_name)
            actual_version = metadata.version(distribution)
        except Exception as exc:
            errors.append(f"{distribution} import failed: {exc}")
            continue
        package_versions[distribution] = actual_version
        if actual_version != expected_version:
            errors.append(
                f"{distribution} {actual_version} != {expected_version}"
            )

    environment = {
        name: os.environ.get(name) for name in CONTROLLED_ENVIRONMENT
    }
    if args.require_controls:
        for name, expected_value in CONTROLLED_ENVIRONMENT.items():
            if environment[name] != expected_value:
                errors.append(
                    f"{name}={environment[name]!r} != {expected_value!r}"
                )

    affinity = None
    try:
        import psutil

        if hasattr(psutil.Process(), "cpu_affinity"):
            affinity = psutil.Process().cpu_affinity()
    except Exception as exc:
        errors.append(f"CPU-affinity inspection failed: {exc}")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "python": actual_python,
        "packages": package_versions,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_affinity": affinity,
        "environment": environment,
        "errors": errors,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
