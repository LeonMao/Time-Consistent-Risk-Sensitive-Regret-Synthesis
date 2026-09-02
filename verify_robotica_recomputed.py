from __future__ import annotations

import csv
from math import isclose, isnan
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
REFERENCE_DIR = ROOT / "06_DATA_AND_RESULTS"
FILE_KEYS = {
    "stage3_3_p0_exact_alpha_m8.csv": ("seed", "planning_alpha"),
    "stage3_3_p0_exact_alpha_m8_summary.csv": ("planning_alpha",),
    "stage3_3_p0_exact_alpha_m10.csv": ("seed", "planning_alpha"),
    "stage3_3_p0_exact_alpha_m10_summary.csv": ("planning_alpha",),
    "stage3_3_p0_full_policy_exactness.csv": ("family", "m", "seed"),
    "stage3_3_p1_hub_m8_plateau_10seeds.csv": ("seed", "alpha"),
    "stage3_3_p1_minimax_plateau_sweep.csv": ("family", "alpha"),
    "stage3_3_p1_minimax_thresholds.csv": ("family", "seed"),
    "stage3_3_p1_operation_counts_raw.csv": ("family", "m", "seed", "method"),
    "stage3_3_p1_operation_counts_summary.csv": ("family", "m", "method"),
    "stage3_3_p1_posterior_workload_raw.csv": ("family", "m", "seed"),
    "stage3_3_p1_posterior_workload_summary.csv": ("family", "m"),
    "stage3_3_p2_3_behavioral_disagreement_raw.csv": ("seed", "p_open", "alpha"),
    "stage3_3_p2_3_behavioral_disagreement_summary.csv": ("alpha", "p_open"),
    "stage3_3_p2_3_cost_vs_regret_raw.csv": ("seed", "p_open", "alpha"),
    "stage3_3_p2_3_cost_vs_regret_summary.csv": ("alpha", "p_open"),
    "stage3_3_p2_3_disagreement_by_alpha.csv": ("alpha",),
}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def sort_rows(rows: list[dict[str, str]], keys: tuple[str, ...]):
    return sorted(rows, key=lambda row: tuple(row[key] for key in keys))


def values_match(expected: str, actual: str) -> bool:
    if expected == actual:
        return True
    if expected == "" or actual == "":
        return False
    try:
        left = float(expected)
        right = float(actual)
    except ValueError:
        return False
    if isnan(left) or isnan(right):
        return isnan(left) and isnan(right)
    return isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def compare_file(reference: Path, reproduced: Path, keys: tuple[str, ...]) -> list[str]:
    failures: list[str] = []
    expected_fields, expected_rows = read_csv(reference)
    actual_fields, actual_rows = read_csv(reproduced)
    if set(expected_fields) != set(actual_fields):
        return [
            f"field mismatch: {reference.name}: "
            f"{sorted(actual_fields)} != {sorted(expected_fields)}"
        ]
    if len(expected_rows) != len(actual_rows):
        return [
            f"row-count mismatch: {reference.name}: "
            f"{len(actual_rows)} != {len(expected_rows)}"
        ]
    expected_rows = sort_rows(expected_rows, keys)
    actual_rows = sort_rows(actual_rows, keys)
    for row_number, (expected, actual) in enumerate(
        zip(expected_rows, actual_rows), start=2
    ):
        for field in expected_fields:
            if not values_match(expected[field], actual[field]):
                failures.append(
                    f"value mismatch: {reference.name}:{row_number}:{field}: "
                    f"{actual[field]!r} != {expected[field]!r}"
                )
                if len(failures) >= 30:
                    return failures
    return failures


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python verify_robotica_recomputed.py STAGE3_3_OUTPUT_DIR")
        return 2
    output_dir = Path(sys.argv[1]).resolve()
    if not output_dir.is_dir():
        print(f"[FAIL] output directory missing: {output_dir}")
        return 1
    actual = {path.name for path in output_dir.iterdir() if path.is_file()}
    expected = set(FILE_KEYS)
    failures = [f"missing output: {name}" for name in sorted(expected - actual)]
    failures.extend(f"unexpected output: {name}" for name in sorted(actual - expected))
    for name, keys in FILE_KEYS.items():
        reproduced = output_dir / name
        if not reproduced.is_file():
            continue
        file_failures = compare_file(
            REFERENCE_DIR / name, reproduced, keys
        )
        failures.extend(file_failures)
        if not file_failures:
            print(f"[PASS:RECOMPUTED] {name}")
    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1
    print(f"ROBOTICA STAGE 3.3 RECOMPUTATION: PASS ({len(expected)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
