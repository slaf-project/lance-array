"""
Benchmark: Zarr 3 (uncompressed + Blosc) vs Lance blob tiles (raw + numcodecs + Blosc2).

Uses :mod:`chunk_tiles` for a shared chunk-aligned API. See README for run instructions.
"""

from __future__ import annotations

import os
import random
import shutil
import sys
import time
import urllib.request
from pathlib import Path

# Repo root on path so `python scripts/lance_vs_zarr/benchmark.py` works without install.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import blosc2
import numcodecs
import numpy as np
import zarr
import zarr.codecs as zc
from numcodecs import Blosc
from PIL import Image
from tqdm import tqdm

from scripts.lance_vs_zarr.chunk_tiles import (
    LanceBlobTileArray,
    ZarrChunkTileArray,
    write_lance_blob_tile_dataset,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
SAMPLE_IMAGE = _SCRIPT_DIR / "sample_2048.jpg"
WORK_DIR = _SCRIPT_DIR / ".bench_out"
ZARR_UNCOMPRESSED_PATH = WORK_DIR / "tiles_uncompressed.zarr"
ZARR_BLOSC_PATH = WORK_DIR / "tiles_blosc.zarr"
LANCE_BLOB_RAW_PATH = WORK_DIR / "tiles_raw.lance"
LANCE_BLOB_BLOSC2_PATH = WORK_DIR / "tiles_blosc2.lance"
LANCE_BLOB_NUMCODECS_PATH = WORK_DIR / "tiles_numcodecs_blosc.lance"
IMAGE_URL = "https://picsum.photos/seed/lance-array-bench/2048/2048"

CHUNK_SHAPE = (256, 256)
N_READS = 500

TYPESIZE = 2
CLEVEL = 5

_NUMCODECS_BLOSC = Blosc(
    cname="zstd",
    clevel=CLEVEL,
    shuffle=Blosc.SHUFFLE,
    typesize=TYPESIZE,
    blocksize=0,
)


def dir_byte_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return total


def ensure_sample_image(path: Path = SAMPLE_IMAGE) -> None:
    if path.exists():
        return
    print(f"Downloading sample image to {path} ...")
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        IMAGE_URL,
        headers={"User-Agent": "lance-array-benchmark/1.0"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        path.write_bytes(resp.read())


def load_grayscale_uint16(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    ensure_sample_image(path)
    im = Image.open(path).convert("L")
    if im.size != (2048, 2048):
        im = im.resize((2048, 2048), Image.Resampling.LANCZOS)
    u8 = np.asarray(im, dtype=np.uint8)
    img = (u8.astype(np.uint32) * 257).astype(np.uint16)
    return img, (img.shape[0], img.shape[1])


# --- per-tile codecs (benchmark-specific; keep separate from chunk_tiles) ---


def encode_uint16_chunk_blosc2(arr: np.ndarray) -> bytes:
    data = np.ascontiguousarray(arr, dtype=np.uint16)
    return blosc2.compress(
        data,
        typesize=TYPESIZE,
        clevel=CLEVEL,
        filter=blosc2.Filter.SHUFFLE,
        codec=blosc2.Codec.ZSTD,
    )


def decode_uint16_chunk_blosc2(blob: bytes, chunk_shape: tuple[int, int]) -> np.ndarray:
    raw = blosc2.decompress(blob)
    expected = chunk_shape[0] * chunk_shape[1] * 2
    if len(raw) != expected:
        raise ValueError(f"decompressed size {len(raw)} != expected {expected}")
    return np.frombuffer(raw, dtype=np.uint16).reshape(chunk_shape)


def encode_uint16_chunk_numcodecs(arr: np.ndarray) -> bytes:
    data = np.ascontiguousarray(arr, dtype=np.uint16)
    return _NUMCODECS_BLOSC.encode(data.tobytes())


def decode_uint16_chunk_numcodecs(
    blob: bytes, chunk_shape: tuple[int, int]
) -> np.ndarray:
    raw = _NUMCODECS_BLOSC.decode(blob)
    expected = chunk_shape[0] * chunk_shape[1] * 2
    if len(raw) != expected:
        raise ValueError(f"decompressed size {len(raw)} != expected {expected}")
    return np.frombuffer(raw, dtype=np.uint16).reshape(chunk_shape)


def encode_uint16_chunk_raw(arr: np.ndarray) -> bytes:
    return np.ascontiguousarray(arr, dtype=np.uint16).tobytes()


def decode_uint16_chunk_raw(blob: bytes, chunk_shape: tuple[int, int]) -> np.ndarray:
    expected = chunk_shape[0] * chunk_shape[1] * 2
    if len(blob) != expected:
        raise ValueError(f"raw blob expected {expected} bytes, got {len(blob)}")
    return np.frombuffer(blob, dtype=np.uint16).reshape(chunk_shape)


def unique_coords_in_order(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for c in coords:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _bench_zarr_reads(
    label: str,
    z_view: ZarrChunkTileArray,
    coords: list[tuple[int, int]],
    unique_coord_list: list[tuple[int, int]],
) -> tuple[float, float]:
    print(f"\nBenchmarking {label} (single chunk reads)...")
    t0 = time.time()
    ch0, ch1 = z_view.chunks
    for i, j in tqdm(coords):
        r0, r1 = i * ch0, (i + 1) * ch0
        c0, c1 = j * ch1, (j + 1) * ch1
        _ = z_view[r0:r1, c0:c1]
    single = time.time() - t0
    print(f"  single: {single:.4f}s  avg {single / N_READS * 1e3:.3f} ms")

    print(f"\nBenchmarking {label} (batched: unique chunks, replay)...")
    t0 = time.time()
    zcache: dict[tuple[int, int], np.ndarray] = {}
    for i, j in unique_coord_list:
        r0, r1 = i * ch0, (i + 1) * ch0
        c0, c1 = j * ch1, (j + 1) * ch1
        zcache[(i, j)] = z_view[r0:r1, c0:c1]
    for i, j in coords:
        _ = zcache[(i, j)]
    batch = time.time() - t0
    print(f"  batched: {batch:.4f}s  avg {batch / N_READS * 1e3:.3f} ms")
    return single, batch


def _bench_lance_blob_reads(
    label: str,
    view: LanceBlobTileArray,
    coords: list[tuple[int, int]],
    row_ids_batch: list[int],
    unique_ids: list[int],
) -> tuple[float, float]:
    ch0, ch1 = view.chunks
    ds = view.dataset

    print(f"\nBenchmarking Lance {label} (single take_blobs + decode)...")
    t0 = time.time()
    for i, j in tqdm(coords):
        r0, r1 = i * ch0, (i + 1) * ch0
        c0, c1 = j * ch1, (j + 1) * ch1
        _ = view[r0:r1, c0:c1]
    single = time.time() - t0
    print(f"  single: {single:.4f}s  avg {single / N_READS * 1e3:.3f} ms")

    print(f"\nBenchmarking Lance {label} (take_blobs batched + decode)...")
    t0 = time.time()
    take_chunk = 64
    blob_cache: dict[int, np.ndarray] = {}
    for k in range(0, len(unique_ids), take_chunk):
        batch = unique_ids[k : k + take_chunk]
        sorted_batch = sorted(batch)
        files = ds.take_blobs(view.blob_column, indices=sorted_batch)
        for rid, f in zip(sorted_batch, files, strict=True):
            blob_cache[rid] = view.decode_blob(f.read())
    for rid in row_ids_batch:
        _ = blob_cache[rid]
    batch = time.time() - t0
    print(f"  batched: {batch:.4f}s  avg {batch / N_READS * 1e3:.3f} ms")
    return single, batch


def main() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading image...")
    img, img_shape = load_grayscale_uint16(SAMPLE_IMAGE)
    assert img_shape == (2048, 2048)
    chunk_shape = CHUNK_SHAPE
    rows = img_shape[0] // chunk_shape[0]
    cols = img_shape[1] // chunk_shape[1]

    for p in (
        ZARR_UNCOMPRESSED_PATH,
        ZARR_BLOSC_PATH,
        LANCE_BLOB_RAW_PATH,
        LANCE_BLOB_BLOSC2_PATH,
        LANCE_BLOB_NUMCODECS_PATH,
    ):
        if p.exists():
            shutil.rmtree(p)

    print(f"Writing Zarr 3 uncompressed (zarr {zarr.__version__})...")
    store_u = zarr.storage.LocalStore(ZARR_UNCOMPRESSED_PATH)
    z_unc = zarr.create_array(
        store_u,
        shape=img_shape,
        chunks=chunk_shape,
        dtype="uint16",
        compressors=[],
        overwrite=True,
    )
    z_view_unc = ZarrChunkTileArray(z_unc)
    z_view_unc.write_numpy(img)
    zarr_unc_disk_mb = dir_byte_size(ZARR_UNCOMPRESSED_PATH) / (1024 * 1024)
    print(f"  Zarr (no compression) on-disk: {zarr_unc_disk_mb:.2f} MiB")

    print("Writing Zarr 3 + Blosc...")
    store_b = zarr.storage.LocalStore(ZARR_BLOSC_PATH)
    blosc_codec = zc.BloscCodec(
        typesize=TYPESIZE,
        cname="zstd",
        clevel=CLEVEL,
        shuffle=zc.BloscShuffle.shuffle,
    )
    z_inner = zarr.create_array(
        store_b,
        shape=img_shape,
        chunks=chunk_shape,
        dtype="uint16",
        compressors=[blosc_codec],
        overwrite=True,
    )
    z_view = ZarrChunkTileArray(z_inner)
    z_view.write_numpy(img)
    zarr_disk_mb = dir_byte_size(ZARR_BLOSC_PATH) / (1024 * 1024)
    print(f"  Zarr (Blosc zstd+shuffle) on-disk: {zarr_disk_mb:.2f} MiB")

    ch0, ch1 = chunk_shape

    def dec_raw(b: bytes) -> np.ndarray:
        return decode_uint16_chunk_raw(b, chunk_shape)

    def dec_b2(b: bytes) -> np.ndarray:
        return decode_uint16_chunk_blosc2(b, chunk_shape)

    def dec_nc(b: bytes) -> np.ndarray:
        return decode_uint16_chunk_numcodecs(b, chunk_shape)

    print("Writing Lance blob v2 + raw uint16...")
    lance_raw = write_lance_blob_tile_dataset(
        LANCE_BLOB_RAW_PATH,
        img,
        chunk_shape,
        encode_uint16_chunk_raw,
        dec_raw,
    )
    mb_raw = dir_byte_size(LANCE_BLOB_RAW_PATH) / (1024 * 1024)
    print(f"  on-disk: {mb_raw:.2f} MiB")

    print(
        f"Writing Lance blob v2 + Blosc2 (python-blosc2 {blosc2.__version__}, "
        f"zstd clevel={CLEVEL}, shuffle, typesize={TYPESIZE})..."
    )
    lance_b2 = write_lance_blob_tile_dataset(
        LANCE_BLOB_BLOSC2_PATH,
        img,
        chunk_shape,
        encode_uint16_chunk_blosc2,
        dec_b2,
    )
    mb_b2 = dir_byte_size(LANCE_BLOB_BLOSC2_PATH) / (1024 * 1024)
    print(f"  on-disk: {mb_b2:.2f} MiB")

    print(
        f"Writing Lance blob v2 + numcodecs Blosc (numcodecs {numcodecs.__version__})..."
    )
    lance_nc = write_lance_blob_tile_dataset(
        LANCE_BLOB_NUMCODECS_PATH,
        img,
        chunk_shape,
        encode_uint16_chunk_numcodecs,
        dec_nc,
    )
    mb_nc = dir_byte_size(LANCE_BLOB_NUMCODECS_PATH) / (1024 * 1024)
    print(f"  on-disk: {mb_nc:.2f} MiB")

    _r = rows // 2 * ch0
    _c = cols // 2 * ch1
    sl = slice(_r, _r + ch0), slice(_c, _c + ch1)
    z_tile = z_view[sl]
    assert np.array_equal(z_tile, z_view_unc[sl])
    assert np.array_equal(z_tile, lance_raw[sl])
    assert np.array_equal(z_tile, lance_nc[sl])
    assert np.array_equal(z_tile, lance_b2[sl])

    coords = [(random.randrange(rows), random.randrange(cols)) for _ in range(N_READS)]
    unique_coord_list = unique_coords_in_order(coords)

    zarr_unc_single, zarr_unc_batch = _bench_zarr_reads(
        "Zarr uncompressed", z_view_unc, coords, unique_coord_list
    )
    zarr_single, zarr_batch = _bench_zarr_reads(
        "Zarr Blosc", z_view, coords, unique_coord_list
    )

    coord_to_row = lance_raw.coord_to_row
    row_ids_batch = [coord_to_row[c] for c in coords]
    unique_ids: list[int] = []
    seen: set[int] = set()
    for rid in row_ids_batch:
        if rid not in seen:
            seen.add(rid)
            unique_ids.append(rid)

    raw_single, raw_batch = _bench_lance_blob_reads(
        "raw uint16 blobs", lance_raw, coords, row_ids_batch, unique_ids
    )
    nc_single, nc_batch = _bench_lance_blob_reads(
        "numcodecs Blosc blobs", lance_nc, coords, row_ids_batch, unique_ids
    )
    b2_single, b2_batch = _bench_lance_blob_reads(
        "Blosc2 blobs", lance_b2, coords, row_ids_batch, unique_ids
    )

    print("\n========== SUMMARY (real photo, uint16) ==========")
    print(f"Unique chunks touched: {len(unique_coord_list)} / {rows * cols}")
    print(
        f"Zarr uncompressed:       {zarr_unc_disk_mb:.2f} MiB  |  single {zarr_unc_single / N_READS * 1e3:.3f} ms  |  batched {zarr_unc_batch / N_READS * 1e3:.3f} ms"
    )
    print(
        f"Zarr Blosc:              {zarr_disk_mb:.2f} MiB  |  single {zarr_single / N_READS * 1e3:.3f} ms  |  batched {zarr_batch / N_READS * 1e3:.3f} ms"
    )
    print(
        f"Lance raw blobs:         {mb_raw:.2f} MiB  |  single {raw_single / N_READS * 1e3:.3f} ms  |  batched {raw_batch / N_READS * 1e3:.3f} ms"
    )
    print(
        f"Lance numcodecs Blosc:   {mb_nc:.2f} MiB  |  single {nc_single / N_READS * 1e3:.3f} ms  |  batched {nc_batch / N_READS * 1e3:.3f} ms"
    )
    print(
        f"Lance Blosc2:            {mb_b2:.2f} MiB  |  single {b2_single / N_READS * 1e3:.3f} ms  |  batched {b2_batch / N_READS * 1e3:.3f} ms"
    )


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as e:
        print(f"Network error (need sample image or working URL): {e}", file=sys.stderr)
        sys.exit(1)
