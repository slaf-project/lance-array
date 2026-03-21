"""
Time random chunk reads: Zarr vs Lance (same pattern as ``create_benchmark_datasets``).

Requires ``scripts/test.zarr`` and ``scripts/test.lance`` from
``create_benchmark_datasets.py``. Use ``--full`` after ``create_benchmark_datasets.py --full``
for the five-way comparison (README order: Zarr/Lance uncompressed, Zarr/Lance Blosc, Lance Blosc2).

On Python 3.11+ with ``zarrs`` installed (``uv sync --extra zarr``), Zarr reads use the
`zarrs <https://github.com/zarrs/zarrs-python>`_ Rust codec pipeline by default. Pass
``--no-zarrs`` for zarr-python's default pipeline, or ``--zarrs`` to force it when auto
would skip (e.g. comparing installs).
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
import zarr
from tqdm import tqdm

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

N_READS = 500


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
    print(f"  single: {single:.4f}s  avg {single / N_READS * 1e3:.3f} ms")

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
    print(f"  batched: {batch:.4f}s  avg {batch / N_READS * 1e3:.3f} ms")
    return single, batch


def bench_lance_reads(
    label: str,
    view: LanceArray,
    coords: list[tuple[int, int]],
    row_ids_batch: list[int],
    unique_ids: list[int],
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
    print(f"  single: {single:.4f}s  avg {single / N_READS * 1e3:.3f} ms")

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
    print(f"  batched: {batch:.4f}s  avg {batch / N_READS * 1e3:.3f} ms")
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
    coords = [(random.randrange(rows), random.randrange(cols)) for _ in range(N_READS)]
    unique_coord_list = unique_coords_in_order(coords)

    if args.full:
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

        z_unc = zarr.open_array(ZARR_UNCOMPRESSED_PATH, mode="r")
        z_blosc = zarr.open_array(ZARR_BLOSC_PATH, mode="r")
        l_raw = open_array(LANCE_BLOB_RAW_PATH)
        l_nc = open_array(LANCE_BLOB_NUMCODECS_PATH)
        l_b2 = open_array(LANCE_BLOB_BLOSC2_PATH)

        row_ids, unique_ids = _row_ids_for_coords(l_raw, coords)

        # Same order as README.md results table: Zarr/Lance pairs, then Lance Blosc2
        zarr_unc_single, zarr_unc_batch = bench_zarr_reads(
            "Zarr (no compression)", z_unc, coords, unique_coord_list
        )
        raw_single, raw_batch = bench_lance_reads(
            "Lance (no compression)", l_raw, coords, row_ids, unique_ids
        )
        zarr_blosc_single, zarr_blosc_batch = bench_zarr_reads(
            "Zarr (numcodecs Blosc)", z_blosc, coords, unique_coord_list
        )
        nc_single, nc_batch = bench_lance_reads(
            "Lance (numcodecs Blosc)", l_nc, coords, row_ids, unique_ids
        )
        b2_single, b2_batch = bench_lance_reads(
            "Lance (Blosc2)", l_b2, coords, row_ids, unique_ids
        )

        zarr_unc_disk_mb = dir_byte_size(ZARR_UNCOMPRESSED_PATH) / (1024 * 1024)
        zarr_disk_mb = dir_byte_size(ZARR_BLOSC_PATH) / (1024 * 1024)
        mb_raw = dir_byte_size(LANCE_BLOB_RAW_PATH) / (1024 * 1024)
        mb_nc = dir_byte_size(LANCE_BLOB_NUMCODECS_PATH) / (1024 * 1024)
        mb_b2 = dir_byte_size(LANCE_BLOB_BLOSC2_PATH) / (1024 * 1024)

        print("\n========== SUMMARY (full suite, .bench_out) ==========")
        print(f"Unique chunks touched: {len(unique_coord_list)} / {rows * cols}")
        print(
            f"Zarr (no compression):   {zarr_unc_disk_mb:.2f} MiB  |  single "
            f"{zarr_unc_single / N_READS * 1e3:.3f} ms  |  batched {zarr_unc_batch / N_READS * 1e3:.3f} ms"
        )
        print(
            f"Lance (no compression):  {mb_raw:.2f} MiB  |  single {raw_single / N_READS * 1e3:.3f} ms  "
            f"|  batched {raw_batch / N_READS * 1e3:.3f} ms"
        )
        print(
            f"Zarr (numcodecs Blosc):  {zarr_disk_mb:.2f} MiB  |  single "
            f"{zarr_blosc_single / N_READS * 1e3:.3f} ms  |  batched {zarr_blosc_batch / N_READS * 1e3:.3f} ms"
        )
        print(
            f"Lance (numcodecs Blosc): {mb_nc:.2f} MiB  |  single {nc_single / N_READS * 1e3:.3f} ms  "
            f"|  batched {nc_batch / N_READS * 1e3:.3f} ms"
        )
        print(
            f"Lance (Blosc2):          {mb_b2:.2f} MiB  |  single {b2_single / N_READS * 1e3:.3f} ms  "
            f"|  batched {b2_batch / N_READS * 1e3:.3f} ms"
        )
        return

    if not TEST_ZARR.exists() or not TEST_LANCE.exists():
        print(
            "Missing test.zarr or test.lance. Run:\n"
            "  uv run python scripts/create_benchmark_datasets.py",
            file=sys.stderr,
        )
        sys.exit(1)

    z_test = zarr.open_array(TEST_ZARR, mode="r")
    la_test = open_array(TEST_LANCE, mode="r")

    row_ids, unique_ids = _row_ids_for_coords(la_test, coords)
    z_single, z_batch = bench_zarr_reads(
        "Zarr (numcodecs Blosc)", z_test, coords, unique_coord_list
    )
    l_single, l_batch = bench_lance_reads(
        "Lance (numcodecs Blosc)", la_test, coords, row_ids, unique_ids
    )

    z_mb = dir_byte_size(TEST_ZARR) / (1024 * 1024)
    l_mb = dir_byte_size(TEST_LANCE) / (1024 * 1024)

    print("\n========== SUMMARY (canonical test.zarr vs test.lance) ==========")
    print(f"Unique chunks touched: {len(unique_coord_list)} / {rows * cols}")
    print(
        f"Zarr (numcodecs Blosc):  {z_mb:.2f} MiB  |  single {z_single / N_READS * 1e3:.3f} ms  "
        f"|  batched {z_batch / N_READS * 1e3:.3f} ms"
    )
    print(
        f"Lance (numcodecs Blosc): {l_mb:.2f} MiB  |  single {l_single / N_READS * 1e3:.3f} ms  "
        f"|  batched {l_batch / N_READS * 1e3:.3f} ms"
    )


if __name__ == "__main__":
    main()
