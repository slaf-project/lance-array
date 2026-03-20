# lance-array

Chunk-aligned **Zarr 3** and **Lance blob v2** tile helpers (`ZarrChunkTileArray`, `LanceBlobTileArray`) plus an optional benchmark comparing Zarr vs Lance for random tile reads.

## Layout

| Path | Purpose |
|------|---------|
| `scripts/lance_vs_zarr/chunk_tiles.py` | Core API (NumPy round-trip, Lance writer) |
| `scripts/lance_vs_zarr/benchmark.py` | Zarr vs Lance benchmark script |
| `scripts/lance_vs_zarr/README.md` | Benchmark details |

## Quick start

```bash
uv sync --extra dev --extra lance --extra zarr
uv run pytest
uv run python scripts/lance_vs_zarr/benchmark.py
```

## Docs

```bash
uv sync --extra docs
uv run mkdocs serve
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
