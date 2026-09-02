from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CHECKSUM_FILE = ROOT / "SHA256SUMS.txt"
FORBIDDEN_PARTS = {
    ".agents", ".git", ".pytest_cache", ".vscode", "00_final_paper",
    "00_final_submission_candidate", "04_theory_framework_stage2",
    "05_manuscript_revision_history", "07_reports_test_outputs",
    "07_validation_reports", "09_recovered_stage_sources",
    "robotica_submission", "robotica_submission_package", "__pycache__",
    "artifacts", "generated", "reproduced_results", "tmp",
}
FORBIDDEN_ENDINGS = (
    ".aux", ".bib", ".docx", ".log", ".out", ".pdf", ".pptx",
    ".pyc", ".pyo", ".synctex.gz", ".tex", ".zip",
)
FORBIDDEN_FILENAMES = {"generate_method_figures.py"}
REQUIRED_FILES = {
    ".gitignore", ".python-version", "CITATION.cff", "LICENSE", "README.md",
    "REPRODUCIBILITY.md", "ROBOTICA_CLAIM_EVIDENCE.md", "THIRD_PARTY.md",
    "PAPER_VERSION.json", "claim_evidence_manifest.json", "generate_checksums.py",
    "pyproject.toml", "requirements-research.txt", "reproduce.py", "uv.lock",
    "verify_release.py", "verify_reproduced_results.py", "verify_robotica_claims.py",
    "verify_robotica_recomputed.py",
}
EXPECTED_FIGURES = {
    "fig_problem_setting.png",
    "fig_method_framework.png",
    "fig_method_factored_solver.png",
    "fig_stage3_5_pkwts_time_consistency_grayscale.png",
    "fig_stage3_3_exact_alpha_tradeoff.png",
    "fig_stage3_3_p1_minimax_plateau.png",
    "fig_stage1_7_horizon_value.png",
    "fig_stage3_5_intel_lab_topology_grayscale.png",
    "fig_stage3_5_layered_benchmark_grayscale.png",
    "fig_stage3_5_controlled_timing_statistics_grayscale.png",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> int:
    failures: list[str] = []
    if not CHECKSUM_FILE.is_file():
        print("[FAIL] missing SHA256SUMS.txt")
        return 1

    expected: dict[str, str] = {}
    for line_number, line in enumerate(CHECKSUM_FILE.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            digest, relative_name = line.split("  ", 1)
        except ValueError:
            failures.append(f"malformed checksum line: {line_number}")
            continue
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            failures.append(f"unsafe checksum path: {relative_name}")
            continue
        normalized = relative.as_posix()
        if normalized in expected:
            failures.append(f"duplicate checksum path: {normalized}")
            continue
        if len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest):
            failures.append(f"invalid checksum digest: {line_number}")
        expected[normalized] = digest.upper()

    actual_paths = {
        path.relative_to(ROOT).as_posix(): path
        for path in ROOT.rglob("*")
        if path.is_file() and path != CHECKSUM_FILE
    }
    for required in sorted(REQUIRED_FILES):
        if required not in actual_paths:
            failures.append(f"required file missing: {required}")

    expected_names = set(expected)
    actual_names = set(actual_paths)
    failures.extend(f"missing: {name}" for name in sorted(expected_names - actual_names))
    failures.extend(f"unexpected: {name}" for name in sorted(actual_names - expected_names))

    for name, path in sorted(actual_paths.items()):
        relative_parts = tuple(part.lower() for part in path.relative_to(ROOT).parts)
        if any(part in FORBIDDEN_PARTS for part in relative_parts):
            failures.append(f"forbidden path: {name}")
        if path.name.lower().endswith(FORBIDDEN_ENDINGS):
            failures.append(f"forbidden file type: {name}")
        if path.name.lower() in FORBIDDEN_FILENAMES:
            failures.append(f"forbidden method-figure generator: {name}")

    for name in sorted(expected_names & actual_names):
        if sha256(actual_paths[name]) != expected[name]:
            failures.append(f"hash mismatch: {name}")

    figures = {path.name for path in (ROOT / "01_FIGURES").glob("*.png")}
    if figures != EXPECTED_FIGURES:
        failures.append(
            "figure set mismatch: missing="
            + repr(sorted(EXPECTED_FIGURES - figures))
            + ", unexpected="
            + repr(sorted(figures - EXPECTED_FIGURES))
        )

    try:
        manifest = json.loads((ROOT / "claim_evidence_manifest.json").read_text(encoding="utf-8"))
        claim_ids = [claim["id"] for claim in manifest["claims"]]
        if claim_ids != [f"C{index:02d}" for index in range(1, 12)]:
            failures.append(f"claim manifest IDs are incomplete or unordered: {claim_ids}")
        for claim in manifest["claims"]:
            for evidence in claim["evidence"]:
                if not (ROOT / evidence).exists():
                    failures.append(f"claim evidence missing ({claim['id']}): {evidence}")
    except Exception as exc:
        failures.append(f"invalid claim evidence manifest: {exc}")

    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8") if (ROOT / "LICENSE").is_file() else ""
    if "MIT License" not in license_text or "Permission is hereby granted" not in license_text:
        failures.append("LICENSE is not a complete MIT license text")

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1
    print(f"RELEASE VERIFICATION: PASS ({len(expected)} checksummed files, 10 figures, 11 claims)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
