from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import numpy as np
from PIL import Image, ImageChops

from reproduce_all_figures import FIGURES, PROJECT_ROOT


REFERENCE_DIR = PROJECT_ROOT / "01_FIGURES"
MAX_MEAN_ABSOLUTE_ERROR = 5.0
MAX_CHANGED_PIXEL_FRACTION = 0.05


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare reproduced figures with the final manuscript images pixel by pixel."
    )
    parser.add_argument(
        "--generated-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "generated",
    )
    args = parser.parse_args()
    failures = []

    for _, filename in FIGURES:
        generated_path = args.generated_dir.resolve() / filename
        reference_path = REFERENCE_DIR / filename
        if not generated_path.is_file():
            failures.append(f"missing: {generated_path}")
            continue
        with Image.open(generated_path) as generated, Image.open(reference_path) as reference:
            generated_rgb = generated.convert("RGB")
            reference_rgb = reference.convert("RGB")
            if generated_rgb.size != reference_rgb.size:
                failures.append(
                    f"size mismatch: {filename}: {generated_rgb.size} != {reference_rgb.size}"
                )
                continue
            if ImageChops.difference(generated_rgb, reference_rgb).getbbox() is None:
                print(f"[PASS:EXACT] {filename}")
                continue
            generated_array = np.asarray(generated_rgb, dtype=np.int16)
            reference_array = np.asarray(reference_rgb, dtype=np.int16)
            absolute_difference = np.abs(generated_array - reference_array)
            mean_absolute_error = float(absolute_difference.mean())
            changed_fraction = float(np.any(absolute_difference != 0, axis=2).mean())
            if (
                mean_absolute_error > MAX_MEAN_ABSOLUTE_ERROR
                or changed_fraction > MAX_CHANGED_PIXEL_FRACTION
            ):
                failures.append(
                    f"visual mismatch: {filename}: MAE={mean_absolute_error:.3f}, "
                    f"changed={changed_fraction:.3%}"
                )
                continue
        print(
            f"[PASS:RENDERER-TOLERANT] {filename}: "
            f"MAE={mean_absolute_error:.3f}, changed={changed_fraction:.3%}"
        )

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1
    print(f"FINAL FIGURE VERIFICATION: PASS ({len(FIGURES)} figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
