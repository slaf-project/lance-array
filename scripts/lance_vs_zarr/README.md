# Lance vs Zarr chunk benchmark

Compares **Zarr 3** with **no compression** and with the **Blosc** chunk codec to **Lance blob v2** rows where each tile is either **raw `uint16`** bytes or compressed with either:

1. **numcodecs.Blosc** — Blosc1 with `zstd`, byte shuffle, `typesize=2`, matching Zarr’s `BloscCodec` parameters. On-disk blob sizes should track Zarr’s chunk sizes closely (Lance adds table/metadata overhead).
2. **python-blosc2** — `blosc2.compress` with the same *logical* settings (zstd, SHUFFLE, `typesize=2`). Blosc2 uses different internal blocking than Blosc1, so **compressed size and speed often differ** from both Zarr and numcodecs; the high-level `compress()` API does not expose all block-splitting knobs (see [python-blosc2](https://github.com/Blosc/python-blosc2) for SChunk / cparams if you want deeper tuning).

## Files

- `chunk_tiles.py` — **fork-friendly core**: `ZarrChunkTileArray`, `LanceBlobTileArray`, `write_lance_blob_tile_dataset`, `normalize_chunk_slices`. Chunk-aligned 2D indexing; NumPy round-trip via `to_numpy()` / `write_numpy()` on Zarr; Lance reads the same plus `write_lance_blob_tile_dataset` to build the table (writes are not exposed on the view — use the writer).
- `__init__.py` — re-exports the public API for `import scripts.lance_vs_zarr` when the repo root is on `PYTHONPATH` or the project is installed.
- `benchmark.py` — creates datasets with those classes and runs timings (codec helpers live here).
- `sample_2048.jpg` — 2048×2048 photo ([Lorem Picsum](https://picsum.photos) fixed seed). If missing, the script downloads it.
- `.bench_out/` — generated stores (gitignored)

### Minimal fork

Copy `chunk_tiles.py` into a new repo, add tests around the two array classes and the Lance writer, and keep `benchmark.py` only if you still want the comparison script.

## Run

From the **repository root**:

```bash
uv sync --extra dev --extra lance --extra zarr
uv run python scripts/lance_vs_zarr/benchmark.py
```

## What it measures

On-disk size and single vs batched random chunk read latency for each backend.
