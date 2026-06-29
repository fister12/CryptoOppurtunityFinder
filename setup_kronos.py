"""Setup script to download Kronos model code from GitHub.

Run this once before first use:
    python setup_kronos.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

KRONOS_REPO = "https://github.com/shiyu-coder/Kronos.git"
KRONOS_DIR = Path(__file__).parent / "kronos_repo"


def main() -> int:
    if KRONOS_DIR.exists():
        print(f"Kronos repo already exists at {KRONOS_DIR}")
        return 0

    print(f"Cloning Kronos into {KRONOS_DIR}...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", KRONOS_REPO, str(KRONOS_DIR)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Failed to clone: {result.stderr}", file=sys.stderr)
        return 1

    print("Kronos repo cloned successfully.")
    print("Models will be auto-downloaded from HuggingFace on first run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
