from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "SHA256SUMS.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> int:
    paths = sorted(
        (path for path in ROOT.rglob("*") if path.is_file() and path != OUTPUT),
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )
    lines = [
        f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}" for path in paths
    ]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"CHECKSUM GENERATION: PASS ({len(paths)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
