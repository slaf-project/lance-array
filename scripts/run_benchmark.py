"""
Time random chunk reads: Zarr vs Lance (same pattern as ``create_benchmark_datasets``).

Requires ``scripts/test.zarr`` and ``scripts/test.lance`` from
``create_benchmark_datasets.py``. Use ``--full`` after ``create_benchmark_datasets.py --full``
for the five-way comparison (README order: Zarr/Lance uncompressed, Zarr/Lance Blosc, Lance Blosc2).

Pass ``--s3`` to read the same layout from object storage. Uses 100 random chunk
reads per run (local runs use 500). Default prefix is ``s3://slaf-datasets/lance_array/``;
e.g. after uploading ``scripts/.bench_out/*``. Needs
``uv sync --extra cloud`` (``smart-open``, ``s3fs``) and AWS credentials in the usual way.
S3-compatible backends (e.g. Tigris) need a custom endpoint: pass
``--s3-endpoint-url`` (see ``--help``) or set ``AWS_ENDPOINT_URL``.

On Python 3.11+ with ``zarrs`` installed (``uv sync --extra zarr``), Zarr reads use the
`zarrs <https://github.com/zarrs/zarrs-python>`_ Rust codec pipeline by default. Pass
``--no-zarrs`` for zarr-python's default pipeline, or ``--zarrs`` to force it when auto
would skip (e.g. comparing installs).
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
import zarr
from tqdm import tqdm
from zarr.errors import ArrayNotFoundError

from create_benchmark_datasets import (
    CHUNK_SHAPE,
    LANCE_BLOB_BLOSC2_PATH,
    LANCE_BLOB_NUMCODECS_PATH,
    LANCE_BLOB_RAW_PATH,
    TEST_LANCE,
    TEST_ZARR,
    WORK_DIR,
    ZARR_BLOSC_PATH,
    ZARR_UNCOMPRESSED_PATH,
    dir_byte_size,
)
from lance_array import LanceArray, open_array

N_READS_LOCAL = 500
N_READS_S3 = 100

DEFAULT_S3_PREFIX = "s3://slaf-datasets/lance_array"
# Tigris (Fly) global endpoint; use with --s3 when the bucket is not on AWS S3.
S3_ENDPOINT_URL = "https://t3.storage.dev"


def _s3_object_uri(prefix: str, name: str) -> str:
    return f"{prefix.rstrip('/')}/{name.lstrip('/')}"


def _require_s3_dependencies() -> None:
    """Zarr uses fsspec + s3fs for s3://; Lance manifest uses smart-open."""
    try:
        import s3fs  # noqa: F401
    except ImportError as e:
        print(
            "--s3 needs s3fs for Zarr on S3. Install: uv sync --extra cloud\n"
            "  (or: pip install s3fs)",
            file=sys.stderr,
        )
        raise SystemExit(1) from e
    try:
        import smart_open  # noqa: F401
    except ImportError as e:
        print(
            "--s3 needs smart-open for Lance on S3. Install: uv sync --extra cloud\n"
            "  (or: pip install smart-open)",
            file=sys.stderr,
        )
        raise SystemExit(1) from e


def _mb_or_dash(mb: float | None) -> str:
    return f"{mb:.2f}" if mb is not None else "—"


def _zarr_storage_options_s3(endpoint_url: str) -> dict[str, Any]:
    """Pass-through for fsspec/s3fs (matches boto3 client endpoint)."""
    return {"client_kwargs": {"endpoint_url": endpoint_url}}


def _open_zarr_read(
    path: str | Path,
    *,
    s3: bool,
    hint: str,
    storage_options: dict[str, Any] | None = None,
):
    kwargs: dict[str, Any] = {"mode": "r"}
    if storage_options is not None:
        kwargs["storage_options"] = storage_options
    try:
        return zarr.open_array(path, **kwargs)
    except ArrayNotFoundError:
        if s3:
            print(
                f"Could not find a Zarr array at {path!r} ({hint}). "
                "Check --s3-prefix: it should be the directory whose children match "
                "scripts/.bench_out/ (tiles_uncompressed.zarr, tiles_blosc.zarr, …). "
                "For Tigris/MinIO, set --s3-endpoint-url or AWS_ENDPOINT_URL.",
                file=sys.stderr,
            )
        raise


def _zarrs_available() -> bool:
    try:
        import zarrs  # noqa: F401
    except ImportError:
        return False
    return True


def _apply_zarr_codec_pipeline(use_zarrs: bool) -> None:
    """Use zarrs Rust pipeline when requested and installed; else zarr-python default."""
    if not use_zarrs:
        return
    if not _zarrs_available():
        print(
            "Note: zarrs not installed or unsupported on this Python; "
            "using zarr-python default codec pipeline. "
            "Install with: uv sync --extra zarr (Python 3.11+).",
            file=sys.stderr,
        )
        return
    zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
    print("Using zarrs Rust codec pipeline for Zarr reads.", file=sys.stderr)


def unique_coords_in_order(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for c in coords:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def bench_zarr_reads(
    label: str,
    z: zarr.Array,
    coords: list[tuple[int, int]],
    unique_coord_list: list[tuple[int, int]],
    *,
    n_reads: int,
) -> tuple[float, float]:
    print(f"\nBenchmarking {label} (single chunk reads)...")
    t0 = time.time()
    ch0, ch1 = int(z.chunks[0]), int(z.chunks[1])
    zdt = np.dtype(z.dtype)
    for i, j in tqdm(coords):
        r0, r1 = i * ch0, (i + 1) * ch0
        c0, c1 = j * ch1, (j + 1) * ch1
        _ = np.asarray(z[r0:r1, c0:c1], dtype=zdt)
    single = time.time() - t0
    print(f"  single: {single:.4f}s  avg {single / n_reads * 1e3:.3f} ms")

    print(f"\nBenchmarking {label} (batched: unique chunks, replay)...")
    t0 = time.time()
    zcache: dict[tuple[int, int], np.ndarray] = {}
    for i, j in unique_coord_list:
        r0, r1 = i * ch0, (i + 1) * ch0
        c0, c1 = j * ch1, (j + 1) * ch1
        zcache[(i, j)] = np.asarray(z[r0:r1, c0:c1], dtype=zdt)
    for i, j in coords:
        _ = zcache[(i, j)]
    batch = time.time() - t0
    print(f"  batched: {batch:.4f}s  avg {batch / n_reads * 1e3:.3f} ms")
    return single, batch


def bench_lance_reads(
    label: str,
    view: LanceArray,
    coords: list[tuple[int, int]],
    row_ids_batch: list[int],
    unique_ids: list[int],
    *,
    n_reads: int,
) -> tuple[float, float]:
    ch0, ch1 = view.chunks
    ds = view.dataset

    print(f"\nBenchmarking {label} (single slice per chunk)...")
    t0 = time.time()
    for i, j in tqdm(coords):
        r0, r1 = i * ch0, (i + 1) * ch0
        c0, c1 = j * ch1, (j + 1) * ch1
        _ = np.asarray(view[r0:r1, c0:c1], dtype=view.dtype)
    single = time.time() - t0
    print(f"  single: {single:.4f}s  avg {single / n_reads * 1e3:.3f} ms")

    print(f"\nBenchmarking {label} (take_blobs batched + decode)...")
    t0 = time.time()
    take_chunk = 64
    tile_cache: dict[int, np.ndarray] = {}
    for k in range(0, len(unique_ids), take_chunk):
        batch = unique_ids[k : k + take_chunk]
        sorted_batch = sorted(batch)
        files = ds.take_blobs(view.blob_column, indices=sorted_batch)
        for rid, f in zip(sorted_batch, files, strict=True):
            tile_cache[rid] = view.decode_tile(f.read())
    for rid in row_ids_batch:
        _ = tile_cache[rid]
    batch = time.time() - t0
    print(f"  batched: {batch:.4f}s  avg {batch / n_reads * 1e3:.3f} ms")
    return single, batch


def _row_ids_for_coords(
    view: LanceArray, coords: list[tuple[int, int]]
) -> tuple[list[int], list[int]]:
    coord_to_row = view.coord_to_row
    row_ids_batch = [coord_to_row[c] for c in coords]
    unique_ids: list[int] = []
    seen: set[int] = set()
    for rid in row_ids_batch:
        if rid not in seen:
            seen.add(rid)
            unique_ids.append(rid)
    return row_ids_batch, unique_ids


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.set_defaults(use_zarrs=None)
    ap.add_argument(
        "--full",
        action="store_true",
        help=f"Expect a full suite under {WORK_DIR} (from create_benchmark_datasets.py --full)",
    )
    ap.add_argument(
        "--s3",
        action="store_true",
        help=(
            "Read datasets from S3 (--s3-prefix) instead of local scripts/ paths; "
            "same filenames as .bench_out/ or test.zarr & test.lance"
        ),
    )
    ap.add_argument(
        "--s3-prefix",
        default=DEFAULT_S3_PREFIX,
        metavar="URI",
        help=(
            "Prefix for --s3 (default: %(default)s). "
            "Full suite: tiles_*.zarr / tiles_*.lance here; "
            "else: test.zarr and test.lance under this prefix"
        ),
    )
    ap.add_argument(
        "--s3-endpoint-url",
        metavar="URL",
        default=None,
        help=(
            "S3-compatible API base URL for --s3 (not needed for AWS). "
            f"Tigris example: {S3_ENDPOINT_URL}. "
            "Sets AWS_ENDPOINT_URL / AWS_ENDPOINT_URL_S3 for this process and "
            "passes storage options to Zarr. You may export those env vars instead."
        ),
    )
    zg = ap.add_mutually_exclusive_group()
    zg.add_argument(
        "--zarrs",
        dest="use_zarrs",
        action="store_const",
        const=True,
        help="Use zarrs (Rust) codec pipeline for Zarr reads (default: on if zarrs is installed)",
    )
    zg.add_argument(
        "--no-zarrs",
        dest="use_zarrs",
        action="store_const",
        const=False,
        help="Use zarr-python default codec pipeline for Zarr reads",
    )
    args = ap.parse_args()
    n_reads = N_READS_S3 if args.s3 else N_READS_LOCAL

    zarr_s3_opts: dict[str, Any] | None = None
    if args.s3:
        _require_s3_dependencies()
        if args.s3_endpoint_url:
            os.environ["AWS_ENDPOINT_URL"] = args.s3_endpoint_url
            os.environ["AWS_ENDPOINT_URL_S3"] = args.s3_endpoint_url
            zarr_s3_opts = _zarr_storage_options_s3(args.s3_endpoint_url)
        msg = f"S3 mode: prefix={args.s3_prefix.rstrip('/')}/ reads={n_reads}"
        if args.s3_endpoint_url:
            msg += f" endpoint={args.s3_endpoint_url}"
        print(msg, file=sys.stderr)

    if args.use_zarrs is True:
        use_zarrs = True
    elif args.use_zarrs is False:
        use_zarrs = False
    else:
        use_zarrs = _zarrs_available()
    _apply_zarr_codec_pipeline(use_zarrs)

    ch0, ch1 = CHUNK_SHAPE
    img_shape = (2048, 2048)
    rows, cols = img_shape[0] // ch0, img_shape[1] // ch1
    coords = [(random.randrange(rows), random.randrange(cols)) for _ in range(n_reads)]
    unique_coord_list = unique_coords_in_order(coords)

    if args.full:
        if not args.s3:
            required = (
                ZARR_UNCOMPRESSED_PATH,
                ZARR_BLOSC_PATH,
                LANCE_BLOB_RAW_PATH,
                LANCE_BLOB_NUMCODECS_PATH,
                LANCE_BLOB_BLOSC2_PATH,
            )
            missing = [p for p in required if not p.exists()]
            if missing:
                print(
                    "--full: missing extended datasets. Run:\n"
                    "  uv run python scripts/create_benchmark_datasets.py --full",
                    file=sys.stderr,
                )
                sys.exit(1)
            z_unc_path: str | Path = ZARR_UNCOMPRESSED_PATH
            z_blosc_path: str | Path = ZARR_BLOSC_PATH
            l_raw_path: str | Path = LANCE_BLOB_RAW_PATH
            l_nc_path: str | Path = LANCE_BLOB_NUMCODECS_PATH
            l_b2_path: str | Path = LANCE_BLOB_BLOSC2_PATH
        else:
            pre = args.s3_prefix
            z_unc_path = _s3_object_uri(pre, "tiles_uncompressed.zarr")
            z_blosc_path = _s3_object_uri(pre, "tiles_blosc.zarr")
            l_raw_path = _s3_object_uri(pre, "tiles_raw.lance")
            l_nc_path = _s3_object_uri(pre, "tiles_numcodecs_blosc.lance")
            l_b2_path = _s3_object_uri(pre, "tiles_blosc2.lance")

        z_unc = _open_zarr_read(
            z_unc_path,
            s3=args.s3,
            hint="uncompressed Zarr in full suite",
            storage_options=zarr_s3_opts if args.s3 else None,
        )
        z_blosc = _open_zarr_read(
            z_blosc_path,
            s3=args.s3,
            hint="Blosc Zarr in full suite",
            storage_options=zarr_s3_opts if args.s3 else None,
        )
        l_raw = open_array(l_raw_path)
        l_nc = open_array(l_nc_path)
        l_b2 = open_array(l_b2_path)

        row_ids, unique_ids = _row_ids_for_coords(l_raw, coords)

        # Same order as README.md results table: Zarr/Lance pairs, then Lance Blosc2
        zarr_unc_single, zarr_unc_batch = bench_zarr_reads(
            "Zarr (no compression)",
            z_unc,
            coords,
            unique_coord_list,
            n_reads=n_reads,
        )
        raw_single, raw_batch = bench_lance_reads(
            "Lance (no compression)",
            l_raw,
            coords,
            row_ids,
            unique_ids,
            n_reads=n_reads,
        )
        zarr_blosc_single, zarr_blosc_batch = bench_zarr_reads(
            "Zarr (numcodecs Blosc)",
            z_blosc,
            coords,
            unique_coord_list,
            n_reads=n_reads,
        )
        nc_single, nc_batch = bench_lance_reads(
            "Lance (numcodecs Blosc)",
            l_nc,
            coords,
            row_ids,
            unique_ids,
            n_reads=n_reads,
        )
        b2_single, b2_batch = bench_lance_reads(
            "Lance (Blosc2)",
            l_b2,
            coords,
            row_ids,
            unique_ids,
            n_reads=n_reads,
        )

        if args.s3:
            zarr_unc_disk_mb = zarr_disk_mb = mb_raw = mb_nc = mb_b2 = None
        else:
            zarr_unc_disk_mb = dir_byte_size(ZARR_UNCOMPRESSED_PATH) / (1024 * 1024)
            zarr_disk_mb = dir_byte_size(ZARR_BLOSC_PATH) / (1024 * 1024)
            mb_raw = dir_byte_size(LANCE_BLOB_RAW_PATH) / (1024 * 1024)
            mb_nc = dir_byte_size(LANCE_BLOB_NUMCODECS_PATH) / (1024 * 1024)
            mb_b2 = dir_byte_size(LANCE_BLOB_BLOSC2_PATH) / (1024 * 1024)

        full_title = (
            "========== SUMMARY (full suite, S3) =========="
            if args.s3
            else "========== SUMMARY (full suite, .bench_out) =========="
        )
        print(f"\n{full_title}")
        print(f"Unique chunks touched: {len(unique_coord_list)} / {rows * cols}")
        print(
            f"Zarr (no compression):   {_mb_or_dash(zarr_unc_disk_mb)} MiB  |  single "
            f"{zarr_unc_single / n_reads * 1e3:.3f} ms  |  batched {zarr_unc_batch / n_reads * 1e3:.3f} ms"
        )
        print(
            f"Lance (no compression):  {_mb_or_dash(mb_raw)} MiB  |  single {raw_single / n_reads * 1e3:.3f} ms  "
            f"|  batched {raw_batch / n_reads * 1e3:.3f} ms"
        )
        print(
            f"Zarr (numcodecs Blosc):  {_mb_or_dash(zarr_disk_mb)} MiB  |  single "
            f"{zarr_blosc_single / n_reads * 1e3:.3f} ms  |  batched {zarr_blosc_batch / n_reads * 1e3:.3f} ms"
        )
        print(
            f"Lance (numcodecs Blosc): {_mb_or_dash(mb_nc)} MiB  |  single {nc_single / n_reads * 1e3:.3f} ms  "
            f"|  batched {nc_batch / n_reads * 1e3:.3f} ms"
        )
        print(
            f"Lance (Blosc2):          {_mb_or_dash(mb_b2)} MiB  |  single {b2_single / n_reads * 1e3:.3f} ms  "
            f"|  batched {b2_batch / n_reads * 1e3:.3f} ms"
        )
        return

    if args.s3:
        z_test_path = _s3_object_uri(args.s3_prefix, "test.zarr")
        l_test_path = _s3_object_uri(args.s3_prefix, "test.lance")
    else:
        if not TEST_ZARR.exists() or not TEST_LANCE.exists():
            print(
                "Missing test.zarr or test.lance. Run:\n"
                "  uv run python scripts/create_benchmark_datasets.py",
                file=sys.stderr,
            )
            sys.exit(1)
        z_test_path = TEST_ZARR
        l_test_path = TEST_LANCE

    z_test = _open_zarr_read(
        z_test_path,
        s3=args.s3,
        hint="canonical test.zarr",
        storage_options=zarr_s3_opts if args.s3 else None,
    )
    la_test = open_array(l_test_path, mode="r")

    row_ids, unique_ids = _row_ids_for_coords(la_test, coords)
    z_single, z_batch = bench_zarr_reads(
        "Zarr (numcodecs Blosc)",
        z_test,
        coords,
        unique_coord_list,
        n_reads=n_reads,
    )
    l_single, l_batch = bench_lance_reads(
        "Lance (numcodecs Blosc)",
        la_test,
        coords,
        row_ids,
        unique_ids,
        n_reads=n_reads,
    )

    if args.s3:
        z_mb = l_mb = None
    else:
        z_mb = dir_byte_size(TEST_ZARR) / (1024 * 1024)
        l_mb = dir_byte_size(TEST_LANCE) / (1024 * 1024)

    canon_title = (
        "========== SUMMARY (canonical test.zarr vs test.lance, S3) =========="
        if args.s3
        else "========== SUMMARY (canonical test.zarr vs test.lance) =========="
    )
    print(f"\n{canon_title}")
    print(f"Unique chunks touched: {len(unique_coord_list)} / {rows * cols}")
    print(
        f"Zarr (numcodecs Blosc):  {_mb_or_dash(z_mb)} MiB  |  single {z_single / n_reads * 1e3:.3f} ms  "
        f"|  batched {z_batch / n_reads * 1e3:.3f} ms"
    )
    print(
        f"Lance (numcodecs Blosc): {_mb_or_dash(l_mb)} MiB  |  single {l_single / n_reads * 1e3:.3f} ms  "
        f"|  batched {l_batch / n_reads * 1e3:.3f} ms"
    )


if __name__ == "__main__":
    main()
