"""
Lance-backed chunk-aligned 2D tile arrays.

Public API: `LanceArray`, `TileCodec`, `normalize_chunk_slices`, `open_array`.

Install with optional extra ``cloud`` for ``smart_open`` (remote ``s3://`` / ``gs://`` opens),
or ``zarr`` for the benchmark stack (Zarr, Blosc2, etc.).
``pylance``, ``pyarrow``, and ``numcodecs`` are core dependencies of `lance_array.core`.
"""

from .core import (
    LanceArray,
    TileCodec,
    normalize_chunk_slices,
    open_array,
)

__all__ = [
    "LanceArray",
    "TileCodec",
    "normalize_chunk_slices",
    "open_array",
]
