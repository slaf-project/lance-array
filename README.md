# lance-array

Chunk-aligned **2D arrays** on **Lance**: store per-tile payloads (often compressed) in Lance **blob v2** columns, fetch them through the object-store path Lance uses, and read them back as **NumPy** in a Zarr-like way—**one physical chunk at a time**.

## Why this exists

Lance blobs are designed to hold **opaque bytes** and to load efficiently from **object storage**. If each blob is one **compressed array chunk** (same idea as a Zarr chunk), you get a simple mental model: **tabular metadata + chunk payloads in Lance** ≈ **Zarr’s manifest + chunk files**, with Lance handling versioning, filtering on non-blob columns, and unified IO.

This package is a small step in that direction: one row per tile, `(i, j)` grid indices, and a blob column holding encoded chunk bytes. Encoding is limited to built-in `TileCodec` presets (**raw**, **numcodecs Blosc**, **Blosc2**). It does **not** implement a full Zarr-compatible codec pipeline or n-dimensional generality; it focuses on **2D** rasters and a clear comparison point against **Zarr 3** in the included benchmark.

## Reading tiles: Zarr vs `LanceArray`

With **Zarr 3**, you open an array and slice; the library reads the chunks that intersect your slice and returns a NumPy array (arbitrary slice shapes are supported).

```python
import numpy as np
import zarr

z = zarr.open_array("path/to/array", mode="r")
tile = np.asarray(z[0:256, 256:512], dtype=np.uint16)  # any valid slice
```

With **`LanceArray`**, you build (or hold) a view over a Lance dataset where **each row is one tile**. **Reads** use the same slice syntax as NumPy/Zarr: every tile that intersects the window is fetched (batched via `take_blobs`), decoded, and stitched—**partial windows** and **strided** slices are supported.

```python
import numpy as np
import zarr
import lance_array as la
from lance_array import LanceArray, TileCodec

image = np.zeros((2048, 2048), dtype=np.uint16)  # your raster

# Built-in codecs: RAW, BLOSC_NUMCODECS (numcodecs is a core dep), BLOSC2 (needs blosc2 → `zarr` extra)
LanceArray.to_lance(
    "path/to/array.lance",
    image,
    chunk_shape=(256, 256),
    codec=TileCodec.RAW,
)
# Reading mirrors zarr.open_array(..., mode="r") + slicing:
z = zarr.open_array("path/to/array.zarr", mode="r")
view = la.open_array("path/to/array.lance", mode="r")

ch0, ch1 = view.chunks
# zarr: ch0, ch1 = z.chunks

window = view[10:100, 5:200]  # axis-aligned step 1; fetches all overlapping tiles
# zarr: window = np.asarray(z[10:100, 5:200], dtype=z.dtype)

tile = view[0:ch0, ch1 : 2 * ch1]  # single-tile slice
# zarr: tile = np.asarray(z[0:ch0, ch1 : 2 * ch1], dtype=z.dtype)

pixel = view[12, 34]  # int, int → 0-d ndarray
# zarr: pixel = np.asarray(z[12, 34], dtype=z.dtype)

full = view.to_numpy()  # whole raster (one batched read path)
# zarr: full = np.asarray(z[:], dtype=z.dtype)   # or np.asarray(z[...], dtype=z.dtype)

np.asarray(view[0:ch0, 0:ch1], dtype=view.dtype)
# zarr: np.asarray(z[0:ch0, 0:ch1], dtype=z.dtype)
```

**Writes** use Lance [merge insert](https://lance.org/guide/read_and_write/#bulk-update) on the `(i, j)` tile keys. Open with **`mode="r+"`**, assign with **basic** indices only—`int` and **`slice` with step `1`**—matching NumPy broadcasting for the RHS. Fancy integer, boolean mask, and strided-slice **assignment** are not implemented (use `LanceArray.to_lance` to replace a full raster).

```python
rw = la.open_array("path/to/array.lance", mode="r+")
rw[10:100, 5:200] = window  # or a scalar; touches every overlapping tile (read–modify–encode–merge)
```

Each dataset directory includes `lance_array.json` so `la.open_array()` / `LanceArray.open()` can restore shape, chunks, dtype, and codec settings.


## Slicing: vs Zarr / NumPy (2D)

| | **Zarr 3** | **`LanceArray`** |
|---|------------|------------------|
| `int` / `slice` (step **1**) | Yes | **Yes** — overlapping tiles batched via `take_blobs`, then stitched |
| Slice step ≠ 1 | Yes | **Yes** — read a bounding box, then stride in memory (NumPy semantics) |
| `...`, row-only (`view[i]`), `np.ix_` | Yes | **Yes** |
| Fancy integer / boolean (incl. two masks) | Zarr varies | **Yes** — NumPy-style 2D rules; pathological masks can touch many tiles |
| Whole raster | `np.asarray(z[...])` | `view.to_numpy()` or `view[:, :]` |
| Write via `[]` | Yes | **`mode="r+"`** — basic `int` / `slice` (step `1`) only; no fancy/boolean/strided **assignment** |

**Not** a full Zarr implementation: **2D only**. Reads decode every Lance tile that intersects the selected indices (fancy reads may widen the window). Writes merge updated tiles in one batch per assignment; design notes live under [`prds/`](prds/).

## Layout

| Path | Purpose |
|------|---------|
| `lance_array/` | Package; implementation in `core.py`, public names on `lance_array` |
| `prds/` | Product notes (e.g. slice writes / `r+` behavior) |
| `scripts/create_benchmark_datasets.py` | Build `test.zarr` / `test.lance` from the JPG (+ optional `--full` → `.bench_out/`) |
| `scripts/run_benchmark.py` | Time reads against those datasets (optional `--full`) |
| `scripts/sample_2048.jpg` | Sample 2048×2048 image; create script downloads if missing |

## Install

```bash
uv sync --extra dev --extra zarr
uv run pytest
```

## Docs

Published site: [slaf-project.github.io/lance-array](https://slaf-project.github.io/lance-array/)

```bash
uv sync --extra docs
uv run mkdocs serve
```

---

## Benchmark

1. **`create_benchmark_datasets.py`** loads `scripts/sample_2048.jpg` into a **uint16** NumPy raster, then writes **`scripts/test.zarr`** (Zarr 3 + Blosc) and **`scripts/test.lance`** (Lance + numcodecs Blosc)—aligned codecs and chunking for a fair pair. With **`--full`**, it also writes the extra variants under `scripts/.bench_out/` (gitignored) for the full five-way table.

2. **`run_benchmark.py`** times **random single-chunk reads** and a **batched** pattern (prefetch unique chunks, replay the same access order). Default mode compares only `test.zarr` vs `test.lance`; **`--full`** benchmarks everything under `.bench_out/`.

### Run

```bash
uv sync --extra dev --extra zarr
uv run python scripts/create_benchmark_datasets.py
uv run python scripts/run_benchmark.py
# Full comparison (README tables):
uv run python scripts/create_benchmark_datasets.py --full
uv run python scripts/run_benchmark.py --full
```

`test.zarr/`, `test.lance/`, and `.bench_out/` are gitignored (regenerate locally). Numbers below are one representative **`run_benchmark.py --full`** run; your machine and storage layer will differ.

### Environment

| | |
|--|--|
| Date | 2025-03-21 |
| Machine | Apple M1 Max, 32 GB RAM |
| OS | macOS 26.0.1 (Tahoe) |
| Python | 3.12.10 |
| `zarr` | 3.1.5 |
| `lance` (PyPI `pylance`) | 3.0.1 |

### Results (2048×2048 uint16, 256×256 chunks, 500 random reads)

This run touched **64 / 64** unique chunks (the full 8×8 tile grid at least once), so the batched path could prefetch every distinct tile once and then replay references.

**Size and latency** (average ms per read over 500 draws; rows pair comparable Zarr vs Lance setups)

| Backend | Size (MiB) | Single-chunk | Batched replay |
|---------|------------|--------------|----------------|
| Zarr (no compression) | 8.00 | 0.654 | 0.147 |
| Lance (no compression) | 8.01 | 0.317 | 0.007 |
| Zarr (numcodecs Blosc) | 1.77 | 0.938 | 0.100 |
| Lance (numcodecs Blosc) | 1.78 | 0.742 | 0.026 |
| Lance (Blosc2) | 1.78 | 0.453 | 0.016 |

**What this is measuring.** “Single” times one slice (`z[i:j, k:l]` vs `view[i:j, k:l]`) per iteration—no cross-iteration caching. “Batched” first loads every **unique** chunk touched by the 500 draws (for Lance, via `take_blobs` in chunks of 64), then replays the same sequence from an in-memory dict; for Zarr, the analogous pattern materializes unique chunks into a Python cache then replays. That batch step is meant to approximate a loader that pays object-store or IPC round trips once per chunk, then serves many consumers from RAM.

**How to read the table.**

- **Size:** The first two rows are the uncompressed pair (~8 MiB); the next three are compressed (~1.77–1.78 MiB on this raster).
- **Single reads:** On this **local SSD** setup, Lance raw is fastest per slice (no decompression, blob read + reshape). Adding Blosc decompression costs CPU on both sides; Lance + numcodecs Blosc is in the same ballpark as Zarr + Blosc. Lance + Blosc2 sits between raw and numcodecs on average here—decoder and framing differ from Blosc1.
- **Batched replay:** Lance pulls ahead strongly because `take_blobs` can fetch many row payloads in one call and the replay phase is pure decode + dict lookup. Zarr still benefits from caching (batched is faster than single) but the relative gain is smaller on a fast local store with many small chunk files.
- **Caveats:** This is not a remote object-store benchmark; latency profiles change when round-trip time dominates. It is also one hardware/OS stack; run the scripts locally before drawing firm conclusions.

## License

Apache-2.0 — see [LICENSE](LICENSE).
