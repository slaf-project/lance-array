"""
Run the full S3 benchmark on a Modal CPU worker.

Requires: ``uv sync --extra modal``, ``modal token new``, and a Modal secret named
**``s3-credentials``** (attached below) containing Tigris/S3 env, e.g. ``AWS_ACCESS_KEY_ID``,
``AWS_SECRET_ACCESS_KEY``, and typically ``AWS_ENDPOINT_URL`` / ``AWS_ENDPOINT_URL_S3``.
Create once: ``modal secret create s3-credentials ...`` (see Modal docs).

Optional env (override defaults):

- ``S3_BENCHMARK_PREFIX`` — default ``s3://slaf-datasets/lance_array``
- ``S3_BENCHMARK_ENDPOINT_URL`` — if set, passed as ``--s3-endpoint-url`` (Zarr fsspec +
  explicit endpoint). Omit when ``AWS_ENDPOINT_URL`` alone is enough for boto/s3fs.

Usage::

    modal run scripts/modal_app.py
"""

from __future__ import annotations

from pathlib import Path

import modal

_REPO_ROOT = Path(__file__).resolve().parent.parent

_BENCH_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy>=1.24.0",
        "pylance>=2.0.0",
        "pyarrow>=14.0.0",
        "numcodecs>=0.12.0",
        "zarr>=2.18.0",
        "blosc2>=3.0.0",
        "pillow>=10.0.0",
        "tqdm>=4.0.0",
        "smart-open[s3]>=6.0.0",
        "s3fs>=2024.1.0",
        "zarrs>=0.2.0",
    )
    .add_local_dir(
        _REPO_ROOT / "lance_array", remote_path="/repo/lance_array", copy=True
    )
    .add_local_dir(_REPO_ROOT / "scripts", remote_path="/repo/scripts", copy=True)
)

app = modal.App(name="lance-array-s3-benchmark")

REMOTE_REPO = "/repo"


@app.function(
    image=_BENCH_IMAGE,
    cpu=4.0,
    memory=4096,
    timeout=3600,
    secrets=[modal.Secret.from_name("s3-credentials")],
)
def run_s3_benchmark() -> int:
    """Execute ``scripts/run_benchmark.py --full --s3`` on Modal (100 reads per S3 mode)."""
    import os
    import subprocess
    import sys

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = REMOTE_REPO if not prev else f"{REMOTE_REPO}{os.pathsep}{prev}"

    cmd = [
        sys.executable,
        f"{REMOTE_REPO}/scripts/run_benchmark.py",
        "--full",
        "--s3",
        "--s3-prefix",
        os.environ.get("S3_BENCHMARK_PREFIX", "s3://slaf-datasets/lance_array"),
    ]
    endpoint = os.environ.get("S3_BENCHMARK_ENDPOINT_URL")
    if endpoint:
        cmd.extend(["--s3-endpoint-url", endpoint])

    proc = subprocess.run(cmd, cwd=REMOTE_REPO, env=env, check=False)
    return proc.returncode


@app.local_entrypoint()
def main() -> None:
    rc = run_s3_benchmark.remote()
    if rc != 0:
        raise SystemExit(rc)
