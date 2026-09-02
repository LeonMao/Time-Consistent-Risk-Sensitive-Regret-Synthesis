from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = PROJECT_ROOT / "01_FIGURES"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "generated"
FIGURES = (
    ("fig_stage3_5_fig01_pkwts_grayscale", "fig_stage3_5_pkwts_time_consistency_grayscale.png"),
    ("fig02_exact_alpha_tradeoff", "fig_stage3_3_exact_alpha_tradeoff.png"),
    ("fig03_minimax_plateau", "fig_stage3_3_p1_minimax_plateau.png"),
    ("fig04_horizon_value", "fig_stage1_7_horizon_value.png"),
    ("fig_stage3_5_fig05_intel_grayscale", "fig_stage3_5_intel_lab_topology_grayscale.png"),
    ("fig_stage3_5_fig06_layered_grayscale", "fig_stage3_5_layered_benchmark_grayscale.png"),
    ("fig_stage3_5_fig06_timing_grayscale", "fig_stage3_5_controlled_timing_statistics_grayscale.png"),
)


def normalize_canvas(path: Path, reference: Path) -> None:
    with Image.open(path) as generated_image, Image.open(reference) as reference_image:
        if generated_image.size == reference_image.size:
            return
        if any(
            generated > expected
            for generated, expected in zip(generated_image.size, reference_image.size)
        ):
            raise RuntimeError(
                f"Generated canvas {generated_image.size} exceeds reference {reference_image.size}: {path.name}"
            )
        if any(
            expected - generated > 1
            for generated, expected in zip(generated_image.size, reference_image.size)
        ):
            raise RuntimeError(
                f"Unexpected canvas mismatch {generated_image.size} vs {reference_image.size}: {path.name}"
            )
        converted = generated_image.convert("RGBA")
        background = converted.getpixel((0, 0))
        normalized = Image.new("RGBA", reference_image.size, background)
        normalized.paste(converted, (0, 0))
        normalized.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate exactly the seven image files used by the final manuscript."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for module_name, filename in FIGURES:
        output = output_dir / filename
        module = importlib.import_module(module_name)
        module.main(output)
        normalize_canvas(output, REFERENCE_DIR / filename)
        print(f"[OK] {filename}: {output}")

    print(f"FINAL FIGURE REPRODUCTION: PASS ({len(FIGURES)} figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
