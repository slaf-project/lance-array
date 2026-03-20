"""
Chunk-aligned 2D tiles: Zarr 3 arrays and Lance blob v2 columns.

This package is intentionally small and dependency-light at import time for the core
tile API (``chunk_tiles``). The benchmark script pulls optional deps
(``blosc2``, ``PIL``, etc.).

Forking: copy ``chunk_tiles.py`` (and optionally ``benchmark.py``) into your own repo;
add tests around :class:`ZarrChunkTileArray` and :class:`LanceBlobTileArray`.
"""

from .chunk_tiles import (
    LanceBlobTileArray,
    ZarrChunkTileArray,
    normalize_chunk_slices,
    write_lance_blob_tile_dataset,
)

__all__ = [
    "LanceBlobTileArray",
    "ZarrChunkTileArray",
    "normalize_chunk_slices",
    "write_lance_blob_tile_dataset",
]
