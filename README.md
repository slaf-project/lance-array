# lance-array

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![CI](https://img.shields.io/github/actions/workflow/status/slaf-project/lance-array/ci.yml?branch=main&label=ci)](https://github.com/slaf-project/lance-array/actions)
[![Docs](https://img.shields.io/github/actions/workflow/status/slaf-project/lance-array/docs.yml?branch=main&label=docs)](https://github.com/slaf-project/lance-array/actions)
[![Code style](https://img.shields.io/badge/code%20style-ruff-black.svg)](https://github.com/astral-sh/ruff)

**lance-array** is chunk-aligned **2D arrays** on **Lance**: one row per tile, blob v2 payloads (often compressed), object-store-friendly IO, and **NumPy**-style slicing—**one physical chunk at a time**. Tabular metadata plus chunk bytes in Lance plays the same role as Zarr’s manifest plus chunk files, with Lance handling versioning, filters on non-blob columns, and unified storage.

**Lance blobs** • **Zarr-like reads** • **Built-in codecs** (raw, numcodecs Blosc, Blosc2 via optional `zarr` extra) • **Benchmarks** vs Zarr 3

## 🚀 Key features

- **2D rasters**: fixed `chunk_shape`, `(i, j)` tile keys, `lance_array.json` sidecar for `open_array()` metadata
- **Reads**: NumPy/Zarr-style indexing—overlapping tiles batched (`take_blobs`), decoded and stitched; partial windows and strided slices supported
- **Writes**: `mode="r+"` with basic `int` / `slice` (step 1) assignment and Lance merge-insert on tile keys; fancy/boolean/strided assignment not supported
- **Not** a full Zarr implementation: no n-D generality or arbitrary codec pipeline—see [docs](https://slaf-project.github.io/lance-array/) and [`prds/`](prds/) for design notes

## 📦 Installation

```bash
git clone https://github.com/slaf-project/lance-array.git
cd lance-array
uv sync
```

**Extras**

| Extra | Purpose |
|-------|---------|
| `zarr` | Zarr 3, Blosc2, Pillow — benchmarks and codec parity |
| `dev` | pytest, ruff, coverage, type checks |
| `docs` | MkDocs + Material + mkdocstrings |
| `cloud` | `smart-open` for remote URIs |

Example: `uv sync --extra dev --extra zarr` then `uv run pytest`.

## ⚡ Quick start

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
window = view[10:100, 5:200]       # batched read + stitch
full = view.to_numpy()             # whole raster

rw = la.open_array("path/to/array.lance", mode="r+")
rw[10:100, 5:200] = window         # merge-insert per overlapping tile
```

Reads mirror `zarr.open_array(..., mode="r")` + slicing; writes are intentionally narrower (basic indices only). Side-by-side Zarr snippets and full semantics live in the [documentation](https://slaf-project.github.io/lance-array/).

## 📚 Documentation

- [Published docs](https://slaf-project.github.io/lance-array/)
- Local preview: `uv sync --extra docs` then `uv run mkdocs serve`

## 📊 Benchmark

Scripts under `scripts/` build aligned Zarr 3 and Lance datasets from `scripts/sample_2048.jpg`, then time random single-chunk reads and a batched replay pattern. `test.zarr/`, `test.lance/`, and `.bench_out/` are gitignored.

```bash
uv sync --extra dev --extra zarr
uv run python scripts/create_benchmark_datasets.py
uv run python scripts/run_benchmark.py
# Full five-way table:
uv run python scripts/create_benchmark_datasets.py --full
uv run python scripts/run_benchmark.py --full
```

**Environment** (representative run)

| | |
|--|--|
| Date | 2026-03-21 |
| Machine | Apple M1 Max, 32 GB RAM |
| OS | macOS 26.0.1 (Tahoe) |
| Python | 3.12.10 |
| `zarr` | 3.1.5 |
| `zarrs` ([zarrs-python](https://github.com/zarrs/zarrs-python), Rust codec pipeline) | 0.2.2 |
| `lance` (PyPI `pylance`) | 3.0.1 |

**Results** (2048×2048 `uint16`, 256×256 chunks, 500 random reads; 64/64 unique chunks touched)

| Backend | Size (MiB) | Single-chunk (ms) | Batched replay (ms) |
|---------|------------|--------------|----------------|
| Zarr (no compression) | 8.00 | 0.350 | 0.039 |
| Lance (no compression) | 8.01 | 0.426 | 0.011 |
| Zarr (numcodecs Blosc) | 1.77 | 0.397 | 0.036 |
| Lance (numcodecs Blosc) | 1.78 | 0.452 | 0.024 |
| Lance (Blosc2) | 1.78 | 0.584 | 0.019 |

**Notes.** 
- “Single” is one slice per iteration without cross-iteration caching. 
- “Batched” prefetches unique chunks (Lance: `take_blobs` in batches; Zarr: in-memory cache), then replays the same access order—useful when amortizing object-store round trips. 
- **Zarr** timings use the **zarrs** Rust codec pipeline (default in `run_benchmark.py` when `zarrs` is installed on Python 3.11+; pass `--no-zarrs` for zarr-python’s default pipeline). 
- Figures are from a **local SSD**; remote latency changes the story. Re-run the scripts on your hardware before drawing conclusions.

## 🙏 Acknowledgments

Built with and compared against:

- [Lance](https://lance.org/) — columnar, versioned datasets and efficient blob columns on object storage
- [Zarr](https://zarr.dev/) — chunked, compressed N-D arrays and the mental model this library echoes for reads

## License

Apache-2.0 — see [LICENSE](LICENSE).