from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import sys

from PIL import Image


ROOT = Path(__file__).resolve().parent
REFERENCE_DIR = ROOT / "06_DATA_AND_RESULTS"
EXACT_FILES = (
    "stage3_5_minimal_robust_ranks.csv",
    "stage3_5_prior_robustness.csv",
    "stage3_5_common_objective_baselines.csv",
    "stage3_5_static_nested_mismatch_raw.csv",
    "stage3_5_static_nested_mismatch_summary.csv",
    "stage3_5_controlled_timing_statistics.csv",
    "stage3_5_intel_spec.json",
)
HORIZON_FILE = "stage3_5_intel_horizon_sensitivity.csv"
HORIZON_TIMING_FIELDS = {
    "initialization_median_s",
    "solve_median_s",
    "total_median_s",
    "total_min_s",
    "total_max_s",
}
GENERATED_ONLY_FILES = (
    "fig_stage3_5_controlled_timing_statistics.png",
    "stage3_5_intel_spec_table.tex",
)
EXPECTED_FILES = set(EXACT_FILES) | {HORIZON_FILE} | set(GENERATED_ONLY_FILES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def compare_horizon(reference: Path, reproduced: Path) -> list[str]:
    failures: list[str] = []
    with reference.open(newline="", encoding="utf-8") as stream:
        expected_reader = csv.DictReader(stream)
        expected_fields = expected_reader.fieldnames
        expected_rows = list(expected_reader)
    with reproduced.open(newline="", encoding="utf-8") as stream:
        actual_reader = csv.DictReader(stream)
        actual_fields = actual_reader.fieldnames
        actual_rows = list(actual_reader)

    if actual_fields != expected_fields:
        return [f"header mismatch: {HORIZON_FILE}"]
    if len(actual_rows) != len(expected_rows):
        return [
            f"row-count mismatch: {HORIZON_FILE}: {len(actual_rows)} != {len(expected_rows)}"
        ]

    compared_fields = [
        field for field in expected_fields or [] if field not in HORIZON_TIMING_FIELDS
    ]
    for index, (expected, actual) in enumerate(zip(expected_rows, actual_rows), start=2):
        for field in compared_fields:
            if actual[field] != expected[field]:
                failures.append(
                    f"scientific-field mismatch: {HORIZON_FILE}:{index}:{field}: "
                    f"{actual[field]!r} != {expected[field]!r}"
                )
                if len(failures) >= 20:
                    return failures
    return failures


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python verify_reproduced_results.py OUTPUT_DIR")
        return 2
    output_dir = Path(sys.argv[1]).resolve()
    if not output_dir.is_dir():
        print(f"[FAIL] output directory missing: {output_dir}")
        return 1

    actual_names = {path.name for path in output_dir.iterdir() if path.is_file()}
    failures = []
    failures.extend(
        f"missing output: {name}" for name in sorted(EXPECTED_FILES - actual_names)
    )
    failures.extend(
        f"unexpected output: {name}" for name in sorted(actual_names - EXPECTED_FILES)
    )

    for name in EXACT_FILES:
        reference = REFERENCE_DIR / name
        reproduced = output_dir / name
        if not reproduced.is_file():
            continue
        if reproduced.read_bytes() != reference.read_bytes():
            failures.append(
                f"byte mismatch: {name}: {sha256(reproduced)} != {sha256(reference)}"
            )
        else:
            print(f"[PASS:EXACT] {name}: {sha256(reproduced)}")

    horizon = output_dir / HORIZON_FILE
    if horizon.is_file():
        horizon_failures = compare_horizon(REFERENCE_DIR / HORIZON_FILE, horizon)
        failures.extend(horizon_failures)
        if not horizon_failures:
            print(f"[PASS:SCIENTIFIC-FIELDS] {HORIZON_FILE}")

    timing_figure = output_dir / GENERATED_ONLY_FILES[0]
    if timing_figure.is_file():
        try:
            with Image.open(timing_figure) as image:
                image.verify()
            print(f"[PASS:VALID-IMAGE] {timing_figure.name}")
        except Exception as exc:
            failures.append(f"invalid generated image: {timing_figure.name}: {exc}")

    tex_output = output_dir / GENERATED_ONLY_FILES[1]
    if tex_output.is_file() and not tex_output.read_text(encoding="utf-8").strip():
        failures.append(f"empty generated table: {tex_output.name}")
    elif tex_output.is_file():
        print(f"[PASS:GENERATED] {tex_output.name}")

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1
    print(f"REPRODUCED RESULT VERIFICATION: PASS ({len(EXPECTED_FILES)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
