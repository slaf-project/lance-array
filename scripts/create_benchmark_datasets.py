"""
Build benchmark rasters under ``scripts/`` from ``sample_2048.jpg``.

Writes:

- ``test.zarr`` — Zarr 3, Blosc (zstd + shuffle), same chunking as the Lance set.
- ``test.lance`` — Lance tiles with :class:`~lance_array.TileCodec.BLOSC_NUMCODECS`
  (closest match to Zarr’s Blosc codec).

With ``--full``, also writes the extra variants under ``scripts/.bench_out/`` for the
full comparison table (uncompressed Zarr, raw / Blosc2 Lance blobs).
"""

from __future__ import annotations

import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

import blosc2
import numcodecs
import numpy as np
import zarr
import zarr.codecs as zc
from PIL import Image
from zarr.storage import LocalStore

from lance_array import LanceArray, TileCodec

SCRIPT_DIR = Path(__file__).resolve().parent
SAMPLE_IMAGE = SCRIPT_DIR / "sample_2048.jpg"
TEST_ZARR = SCRIPT_DIR / "test.zarr"
TEST_LANCE = SCRIPT_DIR / "test.lance"
WORK_DIR = SCRIPT_DIR / ".bench_out"
ZARR_UNCOMPRESSED_PATH = WORK_DIR / "tiles_uncompressed.zarr"
ZARR_BLOSC_PATH = WORK_DIR / "tiles_blosc.zarr"
LANCE_BLOB_RAW_PATH = WORK_DIR / "tiles_raw.lance"
LANCE_BLOB_BLOSC2_PATH = WORK_DIR / "tiles_blosc2.lance"
LANCE_BLOB_NUMCODECS_PATH = WORK_DIR / "tiles_numcodecs_blosc.lance"

IMAGE_URL = "https://picsum.photos/seed/lance-array-bench/2048/2048"
CHUNK_SHAPE = (256, 256)
TYPESIZE = 2
# ZSTD: python-blosc2 often keeps chunks uncompressed at clevel <= 5 while numcodecs
# still compresses; 6+ is a closer match for fair on-disk comparison.
CLEVEL = 6


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


def load_grayscale_uint16(path: Path) -> np.ndarray:
    """Load JPG as 2048×2048 uint16 grayscale (same scaling as the old benchmark)."""
    ensure_sample_image(path)
    im = Image.open(path).convert("L")
    if im.size != (2048, 2048):
        im = im.resize((2048, 2048), Image.Resampling.LANCZOS)
    u8 = np.asarray(im, dtype=np.uint8)
    return (u8.astype(np.uint32) * 257).astype(np.uint16)


def _blosc_zarr_codec() -> zc.BloscCodec:
    return zc.BloscCodec(
        typesize=TYPESIZE,
        cname="zstd",
        clevel=CLEVEL,
        shuffle=zc.BloscShuffle.shuffle,
    )


def write_test_zarr(img: np.ndarray, path: Path):
    """Zarr 3 API: ``create_array`` + whole-array assign (same pattern as Zarr docs)."""
    if path.exists():
        shutil.rmtree(path)
    store = LocalStore(path)
    z = zarr.create_array(
        store,
        shape=img.shape,
        chunks=CHUNK_SHAPE,
        dtype="uint16",
        compressors=[_blosc_zarr_codec()],
        overwrite=True,
    )
    z[:] = img
    return z


def write_test_lance(img: np.ndarray, path: Path) -> LanceArray:
    if path.exists():
        shutil.rmtree(path)
    return LanceArray.to_lance(
        path,
        img,
        CHUNK_SHAPE,
        codec=TileCodec.BLOSC_NUMCODECS,
        blosc_typesize=TYPESIZE,
        blosc_clevel=CLEVEL,
        blosc_cname="zstd",
    )


def create_canonical(img: np.ndarray) -> None:
    print(f"Writing {TEST_ZARR} (Zarr 3 + Blosc)...")
    write_test_zarr(img, TEST_ZARR)
    mb = dir_byte_size(TEST_ZARR) / (1024 * 1024)
    print(f"  on-disk: {mb:.2f} MiB")

    print(f"Writing {TEST_LANCE} (Lance + numcodecs Blosc)...")
    write_test_lance(img, TEST_LANCE)
    mb = dir_byte_size(TEST_LANCE) / (1024 * 1024)
    print(f"  on-disk: {mb:.2f} MiB")


def create_full_suite(img: np.ndarray) -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    for p in (
        ZARR_UNCOMPRESSED_PATH,
        ZARR_BLOSC_PATH,
        LANCE_BLOB_RAW_PATH,
        LANCE_BLOB_BLOSC2_PATH,
        LANCE_BLOB_NUMCODECS_PATH,
    ):
        if p.exists():
            shutil.rmtree(p)

    print(f"Writing Zarr uncompressed → {ZARR_UNCOMPRESSED_PATH} ...")
    store_u = LocalStore(ZARR_UNCOMPRESSED_PATH)
    z_unc = zarr.create_array(
        store_u,
        shape=img.shape,
        chunks=CHUNK_SHAPE,
        dtype="uint16",
        compressors=[],
        overwrite=True,
    )
    z_unc[:] = img
    print(f"  on-disk: {dir_byte_size(ZARR_UNCOMPRESSED_PATH) / (1024 * 1024):.2f} MiB")

    print(f"Writing Zarr Blosc → {ZARR_BLOSC_PATH} ...")
    store_b = LocalStore(ZARR_BLOSC_PATH)
    z_inner = zarr.create_array(
        store_b,
        shape=img.shape,
        chunks=CHUNK_SHAPE,
        dtype="uint16",
        compressors=[_blosc_zarr_codec()],
        overwrite=True,
    )
    z_inner[:] = img
    print(f"  on-disk: {dir_byte_size(ZARR_BLOSC_PATH) / (1024 * 1024):.2f} MiB")

    ch0, ch1 = CHUNK_SHAPE
    print(f"Writing Lance raw → {LANCE_BLOB_RAW_PATH} ...")
    LanceArray.to_lance(LANCE_BLOB_RAW_PATH, img, CHUNK_SHAPE, codec=TileCodec.RAW)
    print(f"  on-disk: {dir_byte_size(LANCE_BLOB_RAW_PATH) / (1024 * 1024):.2f} MiB")

    print(
        f"Writing Lance Blosc2 (python-blosc2 {blosc2.__version__}) → "
        f"{LANCE_BLOB_BLOSC2_PATH} ..."
    )
    LanceArray.to_lance(
        LANCE_BLOB_BLOSC2_PATH,
        img,
        CHUNK_SHAPE,
        codec=TileCodec.BLOSC2,
        blosc_typesize=TYPESIZE,
        blosc_clevel=CLEVEL,
        blosc_cname="zstd",
    )
    print(f"  on-disk: {dir_byte_size(LANCE_BLOB_BLOSC2_PATH) / (1024 * 1024):.2f} MiB")

    print(
        f"Writing Lance numcodecs Blosc (numcodecs {numcodecs.__version__}) → "
        f"{LANCE_BLOB_NUMCODECS_PATH} ..."
    )
    LanceArray.to_lance(
        LANCE_BLOB_NUMCODECS_PATH,
        img,
        CHUNK_SHAPE,
        codec=TileCodec.BLOSC_NUMCODECS,
        blosc_typesize=TYPESIZE,
        blosc_clevel=CLEVEL,
        blosc_cname="zstd",
    )
    print(
        f"  on-disk: {dir_byte_size(LANCE_BLOB_NUMCODECS_PATH) / (1024 * 1024):.2f} MiB"
    )

    _r = img.shape[0] // 2 // ch0 * ch0
    _c = img.shape[1] // 2 // ch1 * ch1
    z_tile = np.asarray(
        z_inner[_r : _r + ch0, _c : _c + ch1], dtype=np.dtype(z_inner.dtype)
    )
    assert np.array_equal(
        z_tile,
        np.asarray(z_unc[_r : _r + ch0, _c : _c + ch1], dtype=np.dtype(z_unc.dtype)),
    )
    lance_raw = LanceArray.open(LANCE_BLOB_RAW_PATH)
    lance_nc = LanceArray.open(LANCE_BLOB_NUMCODECS_PATH)
    lance_b2 = LanceArray.open(LANCE_BLOB_BLOSC2_PATH)
    assert np.array_equal(z_tile, lance_raw[_r : _r + ch0, _c : _c + ch1])
    assert np.array_equal(z_tile, lance_nc[_r : _r + ch0, _c : _c + ch1])
    assert np.array_equal(z_tile, lance_b2[_r : _r + ch0, _c : _c + ch1])


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--full",
        action="store_true",
        help="Also write all variants under scripts/.bench_out/ for run_benchmark.py",
    )
    args = p.parse_args()

    print("Loading image → numpy (uint16)...")
    img = load_grayscale_uint16(SAMPLE_IMAGE)
    assert img.shape == (2048, 2048)

    create_canonical(img)

    if args.full:
        print("\n--full: writing extended suite under .bench_out/ ...")
        create_full_suite(img)
        print(
            "\nNote: test.zarr / test.lance are the canonical pair; .bench_out/ adds "
            "uncompressed Zarr, raw Lance, Blosc2 Lance, and a duplicate numcodecs Lance path."
        )


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as e:
        print(f"Network error (need sample image or working URL): {e}", file=sys.stderr)
        sys.exit(1)
