# lance-array

**lance-array** is chunk-aligned **2D arrays** on **Lance**: one row per tile, blob v2 payloads (often compressed), object-store-friendly IO, and **NumPy**-style slicing—**one physical chunk at a time**. Tabular metadata plus chunk bytes in Lance plays the same role as Zarr’s manifest plus chunk files, with Lance handling versioning, filters on non-blob columns, and unified storage.

**Lance blobs** • **Zarr-like reads** • **Built-in codecs** (raw, numcodecs Blosc, Blosc2 via optional `zarr` extra) • **Benchmarks** vs Zarr 3

Lance blobs hold **opaque bytes** and load efficiently from **object storage**. This package uses one row per tile, `(i, j)` keys, and a blob column for encoded chunk bytes. `TileCodec` covers **raw**, **numcodecs Blosc**, and **Blosc2** (via the `zarr` extra). It is **not** a full Zarr codec pipeline or n-dimensional store; it targets **2D** rasters and fair comparison with **Zarr 3** in the benchmark scripts.

## Quick start

```python
import numpy as np
import lance_array as la
from lance_array import LanceArray, TileCodec

image = np.zeros((2048, 2048), dtype=np.uint16)

LanceArray.to_lance(
    "path/to/array.lance",
    image,
    chunk_shape=(256, 256),
    codec=TileCodec.RAW,
)

view = la.open_array("path/to/array.lance", mode="r")
window = view[10:100, 5:200]
full = view.to_numpy()

rw = la.open_array("path/to/array.lance", mode="r+")
rw[10:100, 5:200] = window
```

Each dataset includes `lance_array.json` so `open_array()` / `LanceArray.open()` can restore shape, chunks, dtype, and codec.

## Zarr vs `LanceArray`

**Zarr 3** — open and slice; intersecting chunks are read and returned as NumPy.

```python
import numpy as np
import zarr

z = zarr.open_array("path/to/array", mode="r")
tile = np.asarray(z[0:256, 256:512], dtype=np.uint16)
```

**`LanceArray`** — one row per tile. Reads use NumPy/Zarr-style indexing: overlapping tiles are fetched in batch (`take_blobs`), decoded, and stitched (including partial windows and strided slices).

```python
import numpy as np
import zarr
import lance_array as la
from lance_array import LanceArray, TileCodec

image = np.zeros((2048, 2048), dtype=np.uint16)

# RAW, BLOSC_NUMCODECS (core), BLOSC2 (install blosc2 → use `zarr` extra)
LanceArray.to_lance(
    "path/to/array.lance",
    image,
    chunk_shape=(256, 256),
    codec=TileCodec.RAW,
)

z = zarr.open_array("path/to/array.zarr", mode="r")
view = la.open_array("path/to/array.lance", mode="r")

ch0, ch1 = view.chunks  # same idea as z.chunks

window = view[10:100, 5:200]
# zarr: np.asarray(z[10:100, 5:200], dtype=z.dtype)

tile = view[0:ch0, ch1 : 2 * ch1]
pixel = view[12, 34]  # 0-d ndarray
full = view.to_numpy()
# zarr: np.asarray(z[:], dtype=z.dtype)

np.asarray(view[0:ch0, 0:ch1], dtype=view.dtype)
```

**Writes** use Lance [merge insert](https://lance.org/guide/read_and_write/#bulk-update) on `(i, j)` tile keys. Use **`mode="r+"`** and **basic** indices only (`int` or `slice` with step `1`), with NumPy broadcasting for the RHS. Fancy integer, boolean, and strided **assignment** are not supported (use `LanceArray.to_lance` for a full raster replace).

```python
rw = la.open_array("path/to/array.lance", mode="r+")
rw[10:100, 5:200] = window  # read–modify–encode–merge per overlapping tile
```

## Slicing vs Zarr / NumPy (2D)

| | **Zarr 3** | **`LanceArray`** |
|---|------------|------------------|
| `int` / `slice` (step **1**) | Yes | Yes — `take_blobs`, then stitch |
| Slice step ≠ 1 | Yes | Yes — bounding box read, stride in memory |
| `...`, row-only (`view[i]`), `np.ix_` | Yes | Yes |
| Fancy integer / boolean | Zarr varies | Yes — NumPy-style 2D rules |
| Whole raster | `np.asarray(z[...])` | `view.to_numpy()` or `view[:, :]` |
| Write via `[]` | Yes | `mode="r+"` — basic indices only |

Reads decode every tile that intersects the index (fancy reads may widen the window). Writes batch merge updates per assignment. Design notes: [`prds/`](https://github.com/slaf-project/lance-array/tree/main/prds) on GitHub.

## Repository layout

| Path | Purpose |
|------|---------|
| `lance_array/` | Package; logic in `core.py` |
| `prds/` | Product notes (e.g. slice writes, `r+`) |
| `scripts/create_benchmark_datasets.py` | `test.zarr` / `test.lance` from JPG; `--full` → `.bench_out/` |
| `scripts/run_benchmark.py` | Timed reads; `--full` for all variants |
| `scripts/sample_2048.jpg` | Sample raster; script can fetch if missing |

## Development

```bash
git clone https://github.com/slaf-project/lance-array.git
cd lance-array
uv sync
```

| Extra | Purpose |
|-------|---------|
| `zarr` | Zarr 3, Blosc2, Pillow — benchmarks |
| `dev` | pytest, ruff, coverage, typing |
| `docs` | MkDocs, Material, mkdocstrings |
| `cloud` | `smart-open` |

```bash
uv sync --extra dev --extra zarr
uv run pytest
```

## Building these docs locally

```bash
uv sync --extra docs
uv run mkdocs serve
```

## Benchmark

`create_benchmark_datasets.py` loads `scripts/sample_2048.jpg` as **uint16**, writes **`scripts/test.zarr`** (Zarr 3 + Blosc) and **`scripts/test.lance`** (Lance + numcodecs Blosc). **`--full`** adds variants under `scripts/.bench_out/` (gitignored). `run_benchmark.py` times random single-chunk reads and a batched replay (prefetch unique chunks, replay order); **`--full`** includes every dataset in `.bench_out/`.

```bash
uv sync --extra dev --extra zarr
uv run python scripts/create_benchmark_datasets.py
uv run python scripts/run_benchmark.py
uv run python scripts/create_benchmark_datasets.py --full
uv run python scripts/run_benchmark.py --full
```

`test.zarr/`, `test.lance/`, and `.bench_out/` are gitignored.

### Environment (representative run)

| | |
|--|--|
| Date | 2025-03-21 |
| Machine | Apple M1 Max, 32 GB RAM |
| OS | macOS 26.0.1 (Tahoe) |
| Python | 3.12.10 |
| `zarr` | 3.1.5 |
| `lance` (PyPI `pylance`) | 3.0.1 |

### Results (2048×2048 uint16, 256×256 chunks, 500 random reads)

All 64 tiles in the 8×8 grid were touched at least once (batched path prefetched every distinct tile once).

| Backend | Size (MiB) | Single-chunk | Batched replay |
|---------|------------|--------------|----------------|
| Zarr (no compression) | 8.00 | 0.654 | 0.147 |
| Lance (no compression) | 8.01 | 0.317 | 0.007 |
| Zarr (numcodecs Blosc) | 1.77 | 0.938 | 0.100 |
| Lance (numcodecs Blosc) | 1.78 | 0.742 | 0.026 |
| Lance (Blosc2) | 1.78 | 0.453 | 0.016 |

**Notes.** “Single” is one slice per iteration, no cross-iteration cache. “Batched” prefetches unique chunks (Lance: `take_blobs`; Zarr: in-memory cache), then replays the same sequence—roughly modeling amortized object-store round trips. **Local SSD** numbers; re-run on your stack before generalizing.

## API reference

- [`lance_array.core`](api/core.md) — `LanceArray`, `open_array`, `normalize_chunk_slices`

## Acknowledgments

- [Lance](https://lance.org/) — columnar, versioned datasets and blob columns on object storage
- [Zarr](https://zarr.dev/) — chunked, compressed N-D arrays and the read model this library follows

## License

Apache-2.0 — see [LICENSE](https://github.com/slaf-project/lance-array/blob/main/LICENSE).
