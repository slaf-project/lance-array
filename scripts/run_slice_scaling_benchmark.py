"""
Compatibility wrapper: slice-scaling is now unified in run_benchmark.py.

Use this script exactly as before; it forwards to:

    uv run python scripts/run_benchmark.py --full ...
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    target = script_dir / "run_benchmark.py"
    cmd = [sys.executable, str(target), "--full", *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
