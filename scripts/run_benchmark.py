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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
import zarr
from tqdm import tqdm
from zarr.errors import ArrayNotFoundError
try:
    from rich.console import Console
    from rich.table import Table
except Exception:  # pragma: no cover - optional dependency
    Console = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]

from create_benchmark_datasets import (
    LANCE_BLOB_BLOSC2_PATH,
    LANCE_BLOB_BLOSC2_MORTON_PATH,
    LANCE_BLOB_RAW_PATH,
    LANCE_BLOB_RAW_MORTON_PATH,
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


def _print_table(title: str, columns: list[str], rows: list[list[str]]) -> None:
    if Console is None or Table is None:
        print(f"\n{title}")
        print(" | ".join(columns))
        for row in rows:
            print(" | ".join(row))
        return
    table = Table(title=title, show_lines=False)
    for col in columns:
        table.add_column(col, no_wrap=True, overflow="ellipsis")
    for row in rows:
        table.add_row(*row)
    Console().print(table)


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


@dataclass
class BenchmarkSeries:
    """Pooled per-read (or per-slice-request) latencies in ms across repeats."""

    single_ms: list[float]
    batch_ms: list[float]
    """Per-read ms for square slice NxN (key = N tile spans)."""
    slice_ms: dict[int, list[float]]


def _empty_benchmark_series(slice_spans: tuple[int, ...]) -> BenchmarkSeries:
    return BenchmarkSeries([], [], {n: [] for n in slice_spans})


def _parse_slice_spans(s: str) -> tuple[int, ...]:
    out: list[int] = []
    seen: set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        n = int(part)
        if n < 1:
            raise ValueError(f"slice size must be >= 1, got {n}")
        if n not in seen:
            seen.add(n)
            out.append(n)
    if not out:
        raise ValueError("--slice-sizes must list at least one positive integer")
    return tuple(out)


def _reference_slice_span(slice_spans: tuple[int, ...]) -> int:
    """Span used for 'primary' slice rows in summary table 1 & 2."""
    if 5 in slice_spans:
        return 5
    return max(slice_spans)


def _pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _fmt_dist(values: list[float]) -> str:
    if not values:
        return "n/a"
    mean = float(np.mean(values))
    return (
        f"mean {mean:.3f} ms, p50 {_pct(values, 50):.3f}, "
        f"p95 {_pct(values, 95):.3f}, p99 {_pct(values, 99):.3f}"
    )


def _stats(values: list[float]) -> tuple[float, float, float, float]:
    return (
        float(np.mean(values)),
        _pct(values, 50),
        _pct(values, 95),
        _pct(values, 99),
    )


def _stats_row(values: list[float]) -> list[str]:
    mean, p50, p95, p99 = _stats(values)
    return [f"{mean:.3f} ms", f"{p50:.3f} ms", f"{p95:.3f} ms", f"{p99:.3f} ms"]


def bench_zarr_reads(
    label: str,
    z: zarr.Array,
    coords: list[tuple[int, int]],
    unique_coord_list: list[tuple[int, int]],
    *,
    n_reads: int,
) -> tuple[float, float, list[float], list[float]]:
    """Return (single_s, batch_s, single_samples_ms, batch_samples_ms)."""
    print(f"\nBenchmarking {label} (single chunk reads)...")
    single_samples, batch_samples = _bench_zarr_once(
        z, coords, unique_coord_list, show_progress=True
    )
    single = sum(single_samples) * 1e-3
    print(f"  single: {single:.4f}s  avg {float(np.mean(single_samples)):.3f} ms")

    print(f"\nBenchmarking {label} (batched: unique chunks, replay)...")
    batch = sum(batch_samples) * 1e-3
    print(f"  batched: {batch:.4f}s  avg {float(np.mean(batch_samples)):.3f} ms")
    return single, batch, single_samples, batch_samples


def _slice_coords(
    rows: int, cols: int, *, span_tiles: int, n_reads: int, seed: int
) -> list[tuple[int, int]]:
    """Top-left tile coordinates for square tile-window reads."""
    if span_tiles < 1:
        raise ValueError("span_tiles must be >= 1")
    if rows < span_tiles or cols < span_tiles:
        raise ValueError(
            f"span_tiles={span_tiles} exceeds tile grid {(rows, cols)}; "
            "reduce slice size or increase image dimensions"
        )
    rng = random.Random(seed)
    max_i = rows - span_tiles
    max_j = cols - span_tiles
    return [(rng.randrange(max_i + 1), rng.randrange(max_j + 1)) for _ in range(n_reads)]


def bench_zarr_slice_reads(
    label: str,
    z: zarr.Array,
    coords: list[tuple[int, int]],
    *,
    span_tiles: int,
    n_reads: int,
) -> list[float]:
    print(f"\nBenchmarking {label} (slice_{span_tiles}x{span_tiles})...")
    samples = _bench_zarr_slice_once(
        z, coords, span_tiles=span_tiles, show_progress=True
    )
    dt = sum(samples) * 1e-3
    print(
        f"  slice_{span_tiles}x{span_tiles}: {dt:.4f}s  "
        f"avg {float(np.mean(samples)):.3f} ms"
    )
    return samples


def _bench_zarr_once(
    z: zarr.Array,
    coords: list[tuple[int, int]],
    unique_coord_list: list[tuple[int, int]],
    *,
    show_progress: bool = False,
) -> tuple[list[float], list[float]]:
    """Per-read latencies in ms: independent chunk reads, then batched cache + replay.

    Batched samples are (replay time per read) + (cache-build wall time / n_reads) so the
    mean tracks the old total/n_reads metric while percentiles reflect replay jitter.
    """
    n_reads = len(coords)
    ch0, ch1 = int(z.chunks[0]), int(z.chunks[1])
    zdt = np.dtype(z.dtype)
    coord_iter = tqdm(coords) if show_progress else coords
    single_samples: list[float] = []
    for i, j in coord_iter:
        r0, r1 = i * ch0, (i + 1) * ch0
        c0, c1 = j * ch1, (j + 1) * ch1
        t0 = time.perf_counter()
        _ = np.asarray(z[r0:r1, c0:c1], dtype=zdt)
        single_samples.append((time.perf_counter() - t0) * 1e3)

    t0 = time.perf_counter()
    zcache: dict[tuple[int, int], np.ndarray] = {}
    for i, j in unique_coord_list:
        r0, r1 = i * ch0, (i + 1) * ch0
        c0, c1 = j * ch1, (j + 1) * ch1
        zcache[(i, j)] = np.asarray(z[r0:r1, c0:c1], dtype=zdt)
    load_ms = (time.perf_counter() - t0) * 1e3
    load_per_read = load_ms / n_reads
    batch_samples: list[float] = []
    for i, j in coords:
        t0 = time.perf_counter()
        _ = zcache[(i, j)]
        batch_samples.append((time.perf_counter() - t0) * 1e3 + load_per_read)
    return single_samples, batch_samples


def _bench_zarr_slice_once(
    z: zarr.Array,
    coords: list[tuple[int, int]],
    *,
    span_tiles: int,
    show_progress: bool = False,
) -> list[float]:
    ch0, ch1 = int(z.chunks[0]), int(z.chunks[1])
    zdt = np.dtype(z.dtype)
    coord_iter = tqdm(coords) if show_progress else coords
    samples: list[float] = []
    for i, j in coord_iter:
        r0, r1 = i * ch0, (i + span_tiles) * ch0
        c0, c1 = j * ch1, (j + span_tiles) * ch1
        t0 = time.perf_counter()
        _ = np.asarray(z[r0:r1, c0:c1], dtype=zdt)
        samples.append((time.perf_counter() - t0) * 1e3)
    return samples


def bench_lance_reads(
    label: str,
    view: LanceArray,
    coords: list[tuple[int, int]],
    row_ids_batch: list[int],
    unique_ids: list[int],
    *,
    n_reads: int,
    profile_fetch_decode: bool = False,
) -> tuple[float, float, list[float], list[float]]:
    """Return (single_s, batch_s, single_samples_ms, batch_samples_ms)."""
    print(f"\nBenchmarking {label} (single slice per chunk)...")
    single_samples, batch_samples, prof = _bench_lance_once(
        view,
        coords,
        row_ids_batch,
        unique_ids,
        show_progress=True,
        profile_fetch_decode=profile_fetch_decode,
    )
    single = sum(single_samples) * 1e-3
    batch = sum(batch_samples) * 1e-3
    print(f"  single: {single:.4f}s  avg {float(np.mean(single_samples)):.3f} ms")

    print(f"\nBenchmarking {label} (batched payload fetch + decode)...")
    print(f"  batched: {batch:.4f}s  avg {float(np.mean(batch_samples)):.3f} ms")
    if profile_fetch_decode and prof is not None:
        fetch_ms_tot, decode_ms_tot = prof
        per_read_fetch = fetch_ms_tot / n_reads
        per_read_decode = decode_ms_tot / n_reads
        print(
            "  batched breakdown: "
            f"fetch={per_read_fetch:.3f} ms/read, "
            f"decode={per_read_decode:.3f} ms/read, "
            f"other={(batch * 1e3 / n_reads) - per_read_fetch - per_read_decode:.3f} ms/read"
        )
    return single, batch, single_samples, batch_samples


def bench_lance_slice_reads(
    label: str,
    view: LanceArray,
    coords: list[tuple[int, int]],
    *,
    span_tiles: int,
    n_reads: int,
) -> list[float]:
    print(f"\nBenchmarking {label} (slice_{span_tiles}x{span_tiles})...")
    samples = _bench_lance_slice_once(
        view, coords, span_tiles=span_tiles, show_progress=True
    )
    dt = sum(samples) * 1e-3
    print(
        f"  slice_{span_tiles}x{span_tiles}: {dt:.4f}s  "
        f"avg {float(np.mean(samples)):.3f} ms"
    )
    return samples


def _bench_lance_once(
    view: LanceArray,
    coords: list[tuple[int, int]],
    row_ids_batch: list[int],
    unique_ids: list[int],
    *,
    show_progress: bool = False,
    profile_fetch_decode: bool = False,
) -> tuple[list[float], list[float], tuple[float, float] | None]:
    """Per-read ms; batched adds amortized cache-build time per read (mean matches total/n).

    When profile_fetch_decode is True, returns (fetch_ms_total, decode_ms_total) for logging.
    """
    n_reads = len(coords)
    ch0, ch1 = view.chunks
    ds = view.dataset
    fetch_s = 0.0
    decode_s = 0.0

    coord_iter = tqdm(coords) if show_progress else coords
    single_samples: list[float] = []
    for i, j in coord_iter:
        r0, r1 = i * ch0, (i + 1) * ch0
        c0, c1 = j * ch1, (j + 1) * ch1
        t0 = time.perf_counter()
        _ = np.asarray(view[r0:r1, c0:c1], dtype=view.dtype)
        single_samples.append((time.perf_counter() - t0) * 1e3)

    t0 = time.perf_counter()
    take_chunk = 64
    tile_cache: dict[int, np.ndarray] = {}
    for k in range(0, len(unique_ids), take_chunk):
        batch = unique_ids[k : k + take_chunk]
        sorted_batch = sorted(batch)
        if view.payload_layout == "bytes":
            if profile_fetch_decode:
                tf = time.perf_counter()
                tbl = ds.take(indices=sorted_batch, columns=[view.blob_column])
                fetch_s += time.perf_counter() - tf
                col = tbl[view.blob_column]
                for idx, rid in enumerate(sorted_batch):
                    td = time.perf_counter()
                    tile_cache[rid] = view.decode_tile(col[idx].as_py())
                    decode_s += time.perf_counter() - td
            else:
                tbl = ds.take(indices=sorted_batch, columns=[view.blob_column])
                col = tbl[view.blob_column]
                for idx, rid in enumerate(sorted_batch):
                    tile_cache[rid] = view.decode_tile(col[idx].as_py())
        else:
            if profile_fetch_decode:
                tf = time.perf_counter()
                files = ds.take_blobs(view.blob_column, indices=sorted_batch)
                fetch_s += time.perf_counter() - tf
                for rid, f in zip(sorted_batch, files, strict=True):
                    tf = time.perf_counter()
                    raw = f.read()
                    fetch_s += time.perf_counter() - tf
                    td = time.perf_counter()
                    tile_cache[rid] = view.decode_tile(raw)
                    decode_s += time.perf_counter() - td
            else:
                files = ds.take_blobs(view.blob_column, indices=sorted_batch)
                for rid, f in zip(sorted_batch, files, strict=True):
                    tile_cache[rid] = view.decode_tile(f.read())

    load_ms = (time.perf_counter() - t0) * 1e3
    load_per_read = load_ms / n_reads
    batch_samples: list[float] = []
    for rid in row_ids_batch:
        t0 = time.perf_counter()
        _ = tile_cache[rid]
        batch_samples.append((time.perf_counter() - t0) * 1e3 + load_per_read)

    prof = (
        (fetch_s * 1e3, decode_s * 1e3) if profile_fetch_decode else None
    )
    return single_samples, batch_samples, prof


def _bench_lance_slice_once(
    view: LanceArray,
    coords: list[tuple[int, int]],
    *,
    span_tiles: int,
    show_progress: bool = False,
) -> list[float]:
    ch0, ch1 = view.chunks
    coord_iter = tqdm(coords) if show_progress else coords
    samples: list[float] = []
    for i, j in coord_iter:
        r0, r1 = i * ch0, (i + span_tiles) * ch0
        c0, c1 = j * ch1, (j + span_tiles) * ch1
        t0 = time.perf_counter()
        _ = np.asarray(view[r0:r1, c0:c1], dtype=view.dtype)
        samples.append((time.perf_counter() - t0) * 1e3)
    return samples


def _row_ids_for_coords(
    view: LanceArray, coords: list[tuple[int, int]]
) -> tuple[list[int], list[int]]:
    coord_to_row = view.coord_to_row
    missing = [c for c in coords if c not in coord_to_row]
    if missing:
        sample = missing[0]
        raise ValueError(
            "benchmark coordinates are incompatible with this Lance dataset grid: "
            f"missing tile {sample} (dataset tile grid is "
            f"{view.n_tile_rows}x{view.n_tile_cols})"
        )
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
    ap.add_argument(
        "--profile-fetch-decode",
        action="store_true",
        help=(
            "Print Lance batched-path breakdown into fetch/read vs decode time. "
            "Useful to determine whether S3 latency or codec decode dominates."
        ),
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for pre-generated chunk coordinates (default: %(default)s)",
    )
    ap.add_argument(
        "--repeats",
        type=int,
        default=1,
        help=(
            "Repeat full benchmark this many times with same coords, randomizing backend "
            "execution order each repeat; prints distribution stats when >1."
        ),
    )
    ap.add_argument(
        "--slice-sizes",
        default="1,3,5,7,9",
        metavar="N,N,...",
        help=(
            "Comma-separated tile spans N for NxN chunk-window slices in --full mode "
            "(default: %(default)s). Also drives the Zarr vs Lance compressed slice table."
        ),
    )
    args = ap.parse_args()
    slice_spans = _parse_slice_spans(args.slice_sizes)
    n_reads = N_READS_S3 if args.s3 else N_READS_LOCAL
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

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

    def _build_coords_for_grid(
        rows: int, cols: int, *, slice_spans: tuple[int, ...]
    ) -> tuple[
        list[tuple[int, int]],
        list[tuple[int, int]],
        dict[int, list[tuple[int, int]]],
    ]:
        rng = random.Random(args.seed)
        coords_local = [
            (rng.randrange(rows), rng.randrange(cols)) for _ in range(n_reads)
        ]
        unique_local = unique_coords_in_order(coords_local)
        slice_coords: dict[int, list[tuple[int, int]]] = {}
        for n in slice_spans:
            slice_coords[n] = _slice_coords(
                rows,
                cols,
                span_tiles=n,
                n_reads=n_reads,
                seed=args.seed + 10_000 + n,
            )
        return coords_local, unique_local, slice_coords

    if args.full:
        if not args.s3:
            required = (
                ZARR_UNCOMPRESSED_PATH,
                ZARR_BLOSC_PATH,
                LANCE_BLOB_RAW_PATH,
                LANCE_BLOB_BLOSC2_PATH,
                LANCE_BLOB_RAW_MORTON_PATH,
                LANCE_BLOB_BLOSC2_MORTON_PATH,
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
            l_b2_path: str | Path = LANCE_BLOB_BLOSC2_PATH
        else:
            pre = args.s3_prefix
            z_unc_path = _s3_object_uri(pre, "tiles_uncompressed.zarr")
            z_blosc_path = _s3_object_uri(pre, "tiles_blosc.zarr")
            l_raw_path = _s3_object_uri(pre, "tiles_raw.lance")
            l_b2_path = _s3_object_uri(pre, "tiles_blosc2.lance")
            l_raw_morton_path = _s3_object_uri(pre, "tiles_raw_morton.lance")
            l_b2_morton_path = _s3_object_uri(pre, "tiles_blosc2_morton.lance")

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
        l_b2 = open_array(l_b2_path)
        if not args.s3:
            l_raw_morton = open_array(LANCE_BLOB_RAW_MORTON_PATH)
            l_b2_morton = open_array(LANCE_BLOB_BLOSC2_MORTON_PATH)
        else:
            l_raw_morton = open_array(l_raw_morton_path)
            l_b2_morton = open_array(l_b2_morton_path)

        rows, cols = l_raw.n_tile_rows, l_raw.n_tile_cols
        for name, view in (
            ("Lance (Blosc2)", l_b2),
            ("Lance (raw, Morton)", l_raw_morton),
            ("Lance (Blosc2, Morton)", l_b2_morton),
        ):
            if (view.n_tile_rows, view.n_tile_cols) != (rows, cols):
                raise ValueError(
                    f"tile-grid mismatch in --full suite: {name} has "
                    f"{view.n_tile_rows}x{view.n_tile_cols}, expected {rows}x{cols}. "
                    "Recreate all full datasets with:\n"
                    "  uv run python scripts/create_benchmark_datasets.py --full"
                )
        coords, unique_coord_list, slice_coords_by_span = _build_coords_for_grid(
            rows, cols, slice_spans=slice_spans
        )
        ref_span_full = _reference_slice_span(slice_spans)

        row_ids_raw, unique_ids_raw = _row_ids_for_coords(l_raw, coords)
        row_ids_b2, unique_ids_b2 = _row_ids_for_coords(l_b2, coords)
        row_ids_raw_morton, unique_ids_raw_morton = _row_ids_for_coords(
            l_raw_morton, coords
        )
        row_ids_b2_morton, unique_ids_b2_morton = _row_ids_for_coords(
            l_b2_morton, coords
        )

        if args.repeats > 1:
            runs: dict[str, BenchmarkSeries] = {
                "Zarr (no compression)": _empty_benchmark_series(slice_spans),
                "Zarr (numcodecs Blosc)": _empty_benchmark_series(slice_spans),
                "Lance (no compression)": _empty_benchmark_series(slice_spans),
                "Lance (Blosc2)": _empty_benchmark_series(slice_spans),
                "Lance (no compression, Morton)": _empty_benchmark_series(slice_spans),
                "Lance (Blosc2, Morton)": _empty_benchmark_series(slice_spans),
            }
            for rep in tqdm(range(args.repeats), desc="repeats", unit="rep"):
                order = list(runs.keys())
                random.Random(args.seed + rep + 1).shuffle(order)
                for key in order:
                    if key == "Zarr (no compression)":
                        s_list, b_list = _bench_zarr_once(
                            z_unc, coords, unique_coord_list
                        )
                        for n in slice_spans:
                            sn = _bench_zarr_slice_once(
                                z_unc, slice_coords_by_span[n], span_tiles=n
                            )
                            runs[key].slice_ms[n].extend(sn)
                    elif key == "Lance (no compression)":
                        s_list, b_list, _ = _bench_lance_once(
                            l_raw, coords, row_ids_raw, unique_ids_raw
                        )
                        for n in slice_spans:
                            sn = _bench_lance_slice_once(
                                l_raw, slice_coords_by_span[n], span_tiles=n
                            )
                            runs[key].slice_ms[n].extend(sn)
                    elif key == "Zarr (numcodecs Blosc)":
                        s_list, b_list = _bench_zarr_once(
                            z_blosc, coords, unique_coord_list
                        )
                        for n in slice_spans:
                            sn = _bench_zarr_slice_once(
                                z_blosc, slice_coords_by_span[n], span_tiles=n
                            )
                            runs[key].slice_ms[n].extend(sn)
                    elif key == "Lance (Blosc2)":
                        s_list, b_list, _ = _bench_lance_once(
                            l_b2, coords, row_ids_b2, unique_ids_b2
                        )
                        for n in slice_spans:
                            sn = _bench_lance_slice_once(
                                l_b2, slice_coords_by_span[n], span_tiles=n
                            )
                            runs[key].slice_ms[n].extend(sn)
                    elif key == "Lance (no compression, Morton)":
                        s_list, b_list, _ = _bench_lance_once(
                            l_raw_morton,
                            coords,
                            row_ids_raw_morton,
                            unique_ids_raw_morton,
                        )
                        for n in slice_spans:
                            sn = _bench_lance_slice_once(
                                l_raw_morton, slice_coords_by_span[n], span_tiles=n
                            )
                            runs[key].slice_ms[n].extend(sn)
                    else:
                        s_list, b_list, _ = _bench_lance_once(
                            l_b2_morton,
                            coords,
                            row_ids_b2_morton,
                            unique_ids_b2_morton,
                        )
                        for n in slice_spans:
                            sn = _bench_lance_slice_once(
                                l_b2_morton, slice_coords_by_span[n], span_tiles=n
                            )
                            runs[key].slice_ms[n].extend(sn)
                    runs[key].single_ms.extend(s_list)
                    runs[key].batch_ms.extend(b_list)

            dist_title = (
                f"========== DISTRIBUTION SUMMARY (full suite, repeats={args.repeats}, "
                f"seed={args.seed}, {'S3' if args.s3 else '.bench_out'}) ==========\n"
                "Stats are per-request latencies (ms), pooled over all reads"
                f" × {args.repeats} repeat(s). Batched rows add amortized cache-build time per read."
            )
            ref_slice_label = f"slice {ref_span_full}x{ref_span_full}"
            best_case_rows: list[list[str]] = [
                ["Zarr uncompressed", "single", *_stats_row(runs["Zarr (no compression)"].single_ms)],
                ["Zarr uncompressed", "batched", *_stats_row(runs["Zarr (no compression)"].batch_ms)],
                ["Zarr uncompressed", ref_slice_label, *_stats_row(runs["Zarr (no compression)"].slice_ms[ref_span_full])],
                ["Zarr compressed", "single", *_stats_row(runs["Zarr (numcodecs Blosc)"].single_ms)],
                ["Zarr compressed", "batched", *_stats_row(runs["Zarr (numcodecs Blosc)"].batch_ms)],
                ["Zarr compressed", ref_slice_label, *_stats_row(runs["Zarr (numcodecs Blosc)"].slice_ms[ref_span_full])],
                ["Lance uncompressed (Morton order)", "single", *_stats_row(runs["Lance (no compression, Morton)"].single_ms)],
                ["Lance uncompressed (Morton order)", "batched", *_stats_row(runs["Lance (no compression, Morton)"].batch_ms)],
                ["Lance uncompressed (Morton order)", ref_slice_label, *_stats_row(runs["Lance (no compression, Morton)"].slice_ms[ref_span_full])],
                ["Lance compressed (Blosc2 + Morton)", "single", *_stats_row(runs["Lance (Blosc2, Morton)"].single_ms)],
                ["Lance compressed (Blosc2 + Morton)", "batched", *_stats_row(runs["Lance (Blosc2, Morton)"].batch_ms)],
                ["Lance compressed (Blosc2 + Morton)", ref_slice_label, *_stats_row(runs["Lance (Blosc2, Morton)"].slice_ms[ref_span_full])],
            ]
            _print_table(
                f"{dist_title}\nBest-case Zarr vs Lance (Morton-order Lance vs row-order Zarr; Blosc2 on compressed Lance)\nUnique chunks touched: {len(unique_coord_list)} / {rows * cols}",
                ["Backend", "Request", "Mean", "p50", "p95", "p99"],
                best_case_rows,
            )
            morton_rows: list[list[str]] = [
                ["Lance uncompressed row", *_stats_row(runs["Lance (no compression)"].slice_ms[ref_span_full])],
                ["Lance uncompressed morton", *_stats_row(runs["Lance (no compression, Morton)"].slice_ms[ref_span_full])],
                ["Lance compressed row (Blosc2)", *_stats_row(runs["Lance (Blosc2)"].slice_ms[ref_span_full])],
                ["Lance compressed morton (Blosc2)", *_stats_row(runs["Lance (Blosc2, Morton)"].slice_ms[ref_span_full])],
            ]
            _print_table(
                f"Morton vs Row Ordering ({ref_slice_label} only)",
                ["Backend", "Mean", "p50", "p95", "p99"],
                morton_rows,
            )
            compressed_slice_rows: list[list[str]] = [
                [
                    f"{n}×{n}",
                    *_stats_row(runs["Zarr (numcodecs Blosc)"].slice_ms[n]),
                    *_stats_row(runs["Lance (Blosc2, Morton)"].slice_ms[n]),
                ]
                for n in slice_spans
            ]
            _print_table(
                "Zarr compressed vs Lance compressed (Blosc2 + Morton order) — by slice size",
                [
                    "NxN",
                    "Zarr μ",
                    "Zarr p50",
                    "Zarr p95",
                    "Zarr p99",
                    "Lance μ",
                    "Lance p50",
                    "Lance p95",
                    "Lance p99",
                ],
                compressed_slice_rows,
            )
            return

        # Same order as README.md results table: Zarr/Lance pairs, then Lance Blosc2
        (
            _,
            _,
            z_unc_single_samp,
            z_unc_batch_samp,
        ) = bench_zarr_reads(
            "Zarr (no compression)",
            z_unc,
            coords,
            unique_coord_list,
            n_reads=n_reads,
        )
        _, _, _, _ = bench_lance_reads(
            "Lance (no compression)",
            l_raw,
            coords,
            row_ids_raw,
            unique_ids_raw,
            n_reads=n_reads,
            profile_fetch_decode=args.profile_fetch_decode,
        )
        (
            _,
            _,
            z_blosc_single_samp,
            z_blosc_batch_samp,
        ) = bench_zarr_reads(
            "Zarr (numcodecs Blosc)",
            z_blosc,
            coords,
            unique_coord_list,
            n_reads=n_reads,
        )
        _, _, _, _ = bench_lance_reads(
            "Lance (Blosc2)",
            l_b2,
            coords,
            row_ids_b2,
            unique_ids_b2,
            n_reads=n_reads,
            profile_fetch_decode=args.profile_fetch_decode,
        )
        (
            _,
            _,
            raw_morton_single_samp,
            raw_morton_batch_samp,
        ) = bench_lance_reads(
            "Lance (no compression, Morton order)",
            l_raw_morton,
            coords,
            row_ids_raw_morton,
            unique_ids_raw_morton,
            n_reads=n_reads,
            profile_fetch_decode=args.profile_fetch_decode,
        )
        (
            _,
            _,
            b2_morton_single_samp,
            b2_morton_batch_samp,
        ) = bench_lance_reads(
            "Lance (Blosc2, Morton order)",
            l_b2_morton,
            coords,
            row_ids_b2_morton,
            unique_ids_b2_morton,
            n_reads=n_reads,
            profile_fetch_decode=args.profile_fetch_decode,
        )
        z_unc_slice_t: dict[int, list[float]] = {}
        raw_slice_t: dict[int, list[float]] = {}
        z_blosc_slice_t: dict[int, list[float]] = {}
        b2_slice_t: dict[int, list[float]] = {}
        raw_morton_slice_t: dict[int, list[float]] = {}
        b2_morton_slice_t: dict[int, list[float]] = {}
        for n in slice_spans:
            z_unc_slice_t[n] = bench_zarr_slice_reads(
                "Zarr (no compression)",
                z_unc,
                slice_coords_by_span[n],
                span_tiles=n,
                n_reads=n_reads,
            )
            raw_slice_t[n] = bench_lance_slice_reads(
                "Lance (no compression)",
                l_raw,
                slice_coords_by_span[n],
                span_tiles=n,
                n_reads=n_reads,
            )
            z_blosc_slice_t[n] = bench_zarr_slice_reads(
                "Zarr (numcodecs Blosc)",
                z_blosc,
                slice_coords_by_span[n],
                span_tiles=n,
                n_reads=n_reads,
            )
            b2_slice_t[n] = bench_lance_slice_reads(
                "Lance (Blosc2)",
                l_b2,
                slice_coords_by_span[n],
                span_tiles=n,
                n_reads=n_reads,
            )
            raw_morton_slice_t[n] = bench_lance_slice_reads(
                "Lance (no compression, Morton order)",
                l_raw_morton,
                slice_coords_by_span[n],
                span_tiles=n,
                n_reads=n_reads,
            )
            b2_morton_slice_t[n] = bench_lance_slice_reads(
                "Lance (Blosc2, Morton order)",
                l_b2_morton,
                slice_coords_by_span[n],
                span_tiles=n,
                n_reads=n_reads,
            )

        full_title = (
            "========== SUMMARY (full suite, S3) =========="
            if args.s3
            else "========== SUMMARY (full suite, .bench_out) =========="
        )
        ref_slice_label = f"slice {ref_span_full}x{ref_span_full}"
        dist_note = (
            "Mean / p50 / p95 / p99 are over per-request latencies (ms) across all reads; "
            "batched adds amortized cache-build time per read."
        )
        best_case_rows = [
            ["Zarr uncompressed", "single", *_stats_row(z_unc_single_samp)],
            ["Zarr uncompressed", "batched", *_stats_row(z_unc_batch_samp)],
            ["Zarr uncompressed", ref_slice_label, *_stats_row(z_unc_slice_t[ref_span_full])],
            ["Zarr compressed", "single", *_stats_row(z_blosc_single_samp)],
            ["Zarr compressed", "batched", *_stats_row(z_blosc_batch_samp)],
            ["Zarr compressed", ref_slice_label, *_stats_row(z_blosc_slice_t[ref_span_full])],
            ["Lance uncompressed (Morton order)", "single", *_stats_row(raw_morton_single_samp)],
            ["Lance uncompressed (Morton order)", "batched", *_stats_row(raw_morton_batch_samp)],
            ["Lance uncompressed (Morton order)", ref_slice_label, *_stats_row(raw_morton_slice_t[ref_span_full])],
            ["Lance compressed (Blosc2 + Morton)", "single", *_stats_row(b2_morton_single_samp)],
            ["Lance compressed (Blosc2 + Morton)", "batched", *_stats_row(b2_morton_batch_samp)],
            ["Lance compressed (Blosc2 + Morton)", ref_slice_label, *_stats_row(b2_morton_slice_t[ref_span_full])],
        ]
        _print_table(
            f"{full_title}\n{dist_note}\nBest-case Zarr vs Lance (Morton-order Lance vs row-order Zarr; Blosc2 on compressed Lance)\nUnique chunks touched: {len(unique_coord_list)} / {rows * cols}",
            ["Backend", "Request", "Mean", "p50", "p95", "p99"],
            best_case_rows,
        )
        morton_rows = [
            ["Lance uncompressed row", *_stats_row(raw_slice_t[ref_span_full])],
            ["Lance uncompressed morton", *_stats_row(raw_morton_slice_t[ref_span_full])],
            ["Lance compressed row (Blosc2)", *_stats_row(b2_slice_t[ref_span_full])],
            ["Lance compressed morton (Blosc2)", *_stats_row(b2_morton_slice_t[ref_span_full])],
        ]
        _print_table(
            f"Morton vs Row Ordering ({ref_slice_label} only)",
            ["Backend", "Mean", "p50", "p95", "p99"],
            morton_rows,
        )
        compressed_slice_rows = [
            [
                f"{n}×{n}",
                *_stats_row(z_blosc_slice_t[n]),
                *_stats_row(b2_morton_slice_t[n]),
            ]
            for n in slice_spans
        ]
        _print_table(
            "Zarr compressed vs Lance compressed (Blosc2 + Morton order) — by slice size",
            [
                "NxN",
                "Zarr μ",
                "Zarr p50",
                "Zarr p95",
                "Zarr p99",
                "Lance μ",
                "Lance p50",
                "Lance p95",
                "Lance p99",
            ],
            compressed_slice_rows,
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
    rows, cols = la_test.n_tile_rows, la_test.n_tile_cols
    coords, unique_coord_list, _slice_canon = _build_coords_for_grid(
        rows, cols, slice_spans=(3, 5)
    )
    coords_slice3 = _slice_canon[3]
    coords_slice5 = _slice_canon[5]

    row_ids, unique_ids = _row_ids_for_coords(la_test, coords)
    _, _, z_ss, z_bs = bench_zarr_reads(
        "Zarr (numcodecs Blosc)",
        z_test,
        coords,
        unique_coord_list,
        n_reads=n_reads,
    )
    _, _, l_ss, l_bs = bench_lance_reads(
        "Lance (numcodecs Blosc)",
        la_test,
        coords,
        row_ids,
        unique_ids,
        n_reads=n_reads,
        profile_fetch_decode=args.profile_fetch_decode,
    )
    z_s3 = bench_zarr_slice_reads(
        "Zarr (numcodecs Blosc)",
        z_test,
        coords_slice3,
        span_tiles=3,
        n_reads=n_reads,
    )
    l_s3 = bench_lance_slice_reads(
        "Lance (numcodecs Blosc)",
        la_test,
        coords_slice3,
        span_tiles=3,
        n_reads=n_reads,
    )
    z_s5 = bench_zarr_slice_reads(
        "Zarr (numcodecs Blosc)",
        z_test,
        coords_slice5,
        span_tiles=5,
        n_reads=n_reads,
    )
    l_s5 = bench_lance_slice_reads(
        "Lance (numcodecs Blosc)",
        la_test,
        coords_slice5,
        span_tiles=5,
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
    canon_rows = [
        [
            "Zarr (numcodecs Blosc)",
            f"{_mb_or_dash(z_mb)} MiB",
            f"{float(np.mean(z_ss)):.3f} ms",
            f"{float(np.mean(z_bs)):.3f} ms",
            f"{float(np.mean(z_s3)):.3f} ms",
            f"{float(np.mean(z_s5)):.3f} ms",
        ],
        [
            "Lance (numcodecs Blosc)",
            f"{_mb_or_dash(l_mb)} MiB",
            f"{float(np.mean(l_ss)):.3f} ms",
            f"{float(np.mean(l_bs)):.3f} ms",
            f"{float(np.mean(l_s3)):.3f} ms",
            f"{float(np.mean(l_s5)):.3f} ms",
        ],
    ]
    _print_table(
        f"{canon_title}\nUnique chunks touched: {len(unique_coord_list)} / {rows * cols}",
        ["Backend", "Size", "Single", "Batched", "Slice 3x3", "Slice 5x5"],
        canon_rows,
    )


if __name__ == "__main__":
    main()
