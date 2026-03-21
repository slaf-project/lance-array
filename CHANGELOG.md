# Changelog

## [0.1.0] - (unreleased)

### Added

- Migrated from `kadalai/scripts/lance_vs_zarr`: `chunk_tiles`, benchmark, CI, pre-commit, and MkDocs scaffolding.

### Changed

- Library code lives in the `lance_array` package; `scripts/` contains only the benchmark and its sample image.
- `LanceBlobTileArray` → `LanceArray`; dataset creation is `LanceArray.to_lance(...)` (replaces `write_lance_blob_tile_dataset`). Removed `ZarrChunkTileArray`; benchmarks use Zarr’s native API.
- Renamed implementation module `lance_array.chunk_tiles` → `lance_array.core`.
- :meth:`LanceArray.to_lance` accepts only :class:`TileCodec` presets (`raw`, `blosc_numcodecs`, `blosc2`); custom ``encode_tile`` / ``decode_tile`` were removed.
- ``pylance``, ``pyarrow``, and ``numcodecs`` are core dependencies; the ``lance`` optional extra was removed.
- :class:`LanceArray` supports arbitrary **2D** ``int`` / ``slice`` (step 1) indexing via batched blob reads and stitching.
