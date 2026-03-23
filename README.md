# lance-array

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![CI](https://img.shields.io/github/actions/workflow/status/slaf-project/lance-array/ci.yml?branch=main&label=ci)](https://github.com/slaf-project/lance-array/actions)
[![Docs](https://img.shields.io/github/actions/workflow/status/slaf-project/lance-array/docs.yml?branch=main&label=docs)](https://github.com/slaf-project/lance-array/actions)
[![Code style](https://img.shields.io/badge/code%20style-ruff-black.svg)](https://github.com/astral-sh/ruff)

**lance-array** is chunk-aligned **2D arrays** on **Lance**. 

- One row per tile, payloads are stored in raw or blosc compressed bytes. 
- Chunks are written in Morton ordering, so spatially contiguous chunks are near each other (same as Zarr).
- Object-store-friendly IO, and **NumPy**-style slicing—**one physical chunk at a time**. 
- Tabular metadata plus chunk bytes in Lance plays the same role as Zarr’s manifest plus chunk files.
- Lance handles versioning, eager prefetching, pushdown filtering on non-bytes columns, and unified storage.

**Lance blobs** • **Zarr-like reads** • **Built-in codecs** (raw, numcodecs Blosc, Blosc2 via optional `zarr` extra) • **Benchmarks** vs Zarr 3

## 🚀 Key features

- **2D rasters**: fixed `chunk_shape`, `(i, j)` tile keys, `lance_array.json` sidecar for `open_array()` metadata.
- **Reads**: NumPy/Zarr-style indexing—overlapping tiles batched (`take`), decoded and stitched; partial windows and strided slices supported.
- **Writes**: `mode="r+"` with basic `int` / `slice` (step 1) assignment and Lance merge-insert on tile keys; fancy/boolean/strided assignment not supported.
- **Not** a full Zarr implementation: no n-D generality or arbitrary codec pipeline.

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
| `cloud` | `smart-open[s3]`, `s3fs` — remote URIs and S3 benchmarks |
| `modal` | [Modal](https://modal.com/) — run the S3 benchmark on a remote CPU (`modal run modal_app.py`) |

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

Reads mirror `zarr.open_array(..., mode="r")` + slicing; writes are intentionally narrower (basic indices only).

## 📚 Documentation

- [Quickstart](https://slaf-project.github.io/lance-array/)

## 📊 Benchmark

Scripts under `scripts/` build aligned Zarr 3 and Lance datasets from `scripts/sample_2048.jpg`, then time random single-chunk reads and a batched replay pattern. `test.zarr/`, `test.lance/`, and `.bench_out/` are gitignored. Chunk size is 64 x 64.

```bash
uv sync --extra dev --extra zarr
uv run python scripts/create_benchmark_datasets.py
uv run python scripts/run_benchmark.py
# Full five-way table:
uv run python scripts/create_benchmark_datasets.py --full
uv run python scripts/run_benchmark.py --full
# Same suite on object storage (needs --extra cloud; 100 reads in S3 mode):
# uv run python scripts/run_benchmark.py --full --s3
```


**Environment** (representative run)

| | |
|--|--|
| Date | 2026-03-23 |
| Machine | Apple M1 Max, 32 GB RAM |
| OS | macOS 26.0.1 (Tahoe) |
| Python | 3.12.10 |
| `zarr` | 3.1.5 |
| `zarrs` ([zarrs-python](https://github.com/zarrs/zarrs-python), Rust codec pipeline) | 0.2.2 |
| `lance` (PyPI `pylance`) | 3.0.1 |

### Full-suite latency (p50 / p95 / p99)

The `run_benchmark.py --full` tables report **per-request** latencies. **Means** are easy to skew (e.g. first read / cold cache), so the charts use **p50 / p95 / p99** on a **shared x-axis**; each **horizontal facet** is one condition (single tile uncompressed/compressed, then each slice size). **Zarr** and **Lance** are paired bars per percentile; **y** is comparable across p50–p99 within each facet. Captions for methodology and data source are **below each figure**. Generated from captured benchmark output:

- `scripts/local_summary.txt` — SSD → laptop  
- `scripts/s3_summary.txt` — object store (e.g. Tigris) → laptop  

Regenerate SVGs after updating those files:

```bash
uv sync --extra dev
uv run python scripts/render_benchmark_charts.py
```

**Labels.** **Lance uncompressed (Morton order)** is **raw** payload (no Blosc2)—only **Morton (Z-order) tile sequencing** in the Lance table. **Lance compressed (Blosc2 + Morton)** is Blosc2-compressed tiles with the same Morton ordering.

![Full benchmark local — p50 / p95 / p99 per request](docs/images/benchmarks/benchmark_local_p50_p95_p99.svg)

*Caption — **local SSD → laptop**:* Per-request latency; **means omitted** (often skewed by cold starts). **Batched + replay** not shown. **Single tile (uncompressed):** Zarr row-major chunk order vs Lance **raw** payload and **Morton (Z-order)** tile rows. **Single tile (compressed)** and **slices:** Zarr **numcodecs Blosc** vs Lance **Blosc2** with the same Morton ordering. Slices use every N×N row from the compressed scaling table. Source: `scripts/local_summary.txt`.

![Full benchmark S3 → laptop — p50 / p95 / p99 per request](docs/images/benchmarks/benchmark_s3_p50_p95_p99.svg)

*Caption — **object store → laptop**:* Same layout and comparisons as above. Source: `scripts/s3_summary.txt`.


## 🙏 Acknowledgments

Built with and compared against:

- [Lance](https://lance.org/) — columnar, versioned datasets and efficient blob columns on object storage
- [Zarr](https://zarr.dev/) — chunked, compressed N-D arrays and the mental model this library echoes for reads

## License

Apache-2.0 — see [LICENSE](LICENSE).