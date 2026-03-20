"""
Chunk-aligned 2D tile access for Zarr 3 arrays and Lance blob v2 columns.

Both backends expose the same *tile* addressing model: ``view[r0:r1, c0:c1]`` must
be exactly one physical chunk, aligned to ``chunks``. This matches how many benchmarks
and training loaders access raster data.

**Zarr:** read/write arbitrary aligned chunks via the underlying ``zarr.Array``.

**Lance:** each row holds one encoded tile in a blob column; reads use
``Dataset.take_blobs``. There is no in-place tile update in this helper — use
:func:`write_lance_blob_tile_dataset` to build or replace a dataset from a full
``numpy`` image.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import lance
import numpy as np
import pyarrow as pa
from lance import blob_array, blob_field

__all__ = [
    "LanceBlobTileArray",
    "ZarrChunkTileArray",
    "normalize_chunk_slices",
    "write_lance_blob_tile_dataset",
]


def normalize_chunk_slices(s: slice, dim: int) -> tuple[int, int]:
    """Return ``(start, stop)`` for a slice with step 1 and positive length."""
    start, stop, step = s.indices(dim)
    if step != 1:
        raise ValueError("step must be 1")
    if stop <= start:
        raise ValueError("empty slice not supported")
    return start, stop


class ZarrChunkTileArray:
    """Chunk-aligned view over a 2D Zarr array.

    **Read / write tiles:** ``tile = view[r0:r1, c0:c1]`` and
    ``view[r0:r1, c0:c1] = tile`` where the slice is exactly one chunk and
    chunk-aligned (same rules as :meth:`__getitem__`).

    **Full raster:** :meth:`to_numpy` materializes the whole grid;
    :meth:`write_numpy` writes a full ``(H, W)`` array by chunk.
    """

    def __init__(self, z: Any) -> None:
        """
        Parameters
        ----------
        z
            An opened 2D ``zarr.Array`` (Zarr 3).
        """
        self._z = z
        self.shape: tuple[int, int] = tuple(int(x) for x in z.shape)
        self.chunks: tuple[int, int] = tuple(int(x) for x in z.chunks)
        self.dtype: np.dtype = np.dtype(z.dtype)

    def __getitem__(self, key: tuple[slice, slice]) -> np.ndarray:
        r0, r1 = normalize_chunk_slices(key[0], self.shape[0])
        c0, c1 = normalize_chunk_slices(key[1], self.shape[1])
        ch0, ch1 = self.chunks
        if (r1 - r0) != ch0 or (c1 - c0) != ch1:
            raise ValueError("slice must match one full chunk shape")
        if r0 % ch0 or c0 % ch1:
            raise ValueError("slice must be chunk-aligned")
        return np.asarray(self._z[r0:r1, c0:c1], dtype=self.dtype)

    def __setitem__(self, key: tuple[slice, slice], value: np.ndarray) -> None:
        r0, r1 = normalize_chunk_slices(key[0], self.shape[0])
        c0, c1 = normalize_chunk_slices(key[1], self.shape[1])
        ch0, ch1 = self.chunks
        if (r1 - r0) != ch0 or (c1 - c0) != ch1:
            raise ValueError("slice must match one full chunk shape")
        if r0 % ch0 or c0 % ch1:
            raise ValueError("slice must be chunk-aligned")
        arr = np.asarray(value, dtype=self.dtype)
        if arr.shape != (ch0, ch1):
            raise ValueError(f"value shape {arr.shape} != chunk {ch0, ch1}")
        self._z[r0:r1, c0:c1] = arr

    def to_numpy(self) -> np.ndarray:
        """Assemble the full 2D array by reading every chunk in row-major tile order."""
        h, w = self.shape
        ch0, ch1 = self.chunks
        out = np.empty((h, w), dtype=self.dtype)
        for r0 in range(0, h, ch0):
            for c0 in range(0, w, ch1):
                out[r0 : r0 + ch0, c0 : c0 + ch1] = self[r0 : r0 + ch0, c0 : c0 + ch1]
        return out

    def write_numpy(self, image: np.ndarray) -> None:
        """Write a full ``(H, W)`` raster, one chunk at a time.

        Raises
        ------
        ValueError
            If ``image.shape != self.shape`` or dtype does not match.
        """
        if image.shape != self.shape:
            raise ValueError(f"image shape {image.shape} != {self.shape}")
        if np.dtype(image.dtype) != self.dtype:
            raise ValueError(f"image dtype {image.dtype} != {self.dtype}")
        h, w = self.shape
        ch0, ch1 = self.chunks
        for r0 in range(0, h, ch0):
            for c0 in range(0, w, ch1):
                self[r0 : r0 + ch0, c0 : c0 + ch1] = image[r0 : r0 + ch0, c0 : c0 + ch1]


class LanceBlobTileArray:
    """Read-only chunk-aligned view over Lance rows with one blob tile per row.

    Rows are indexed by logical tile grid ``(tile_i, tile_j)`` mapped to a Lance
    row id. Each blob is decoded with ``decode_tile(bytes) -> (ch0, ch1)`` ndarray.

    **Read tile:** ``tile = view[r0:r1, c0:c1]`` (same chunk rules as Zarr).

    **Full raster:** :meth:`to_numpy` reads and stitches all tiles. Writes are not
    supported on an open dataset; call :func:`write_lance_blob_tile_dataset` to
    create a new dataset from a numpy image.
    """

    def __init__(
        self,
        dataset: lance.LanceDataset,
        chunk_shape: tuple[int, int],
        image_shape: tuple[int, int],
        coord_to_row: dict[tuple[int, int], int],
        decode_tile: Callable[[bytes], np.ndarray],
        *,
        blob_column: str = "blob",
        dtype: np.dtype | None = None,
    ) -> None:
        self._ds = dataset
        self.shape: tuple[int, int] = image_shape
        self.chunks: tuple[int, int] = chunk_shape
        self.dtype: np.dtype = (
            np.dtype(dtype) if dtype is not None else np.dtype(np.uint16)
        )
        self._coord_to_row = coord_to_row
        self._decode_tile = decode_tile
        self._blob_column = blob_column

    @property
    def coord_to_row(self) -> dict[tuple[int, int], int]:
        """Map ``(tile_i, tile_j)`` to Lance row index (for batched ``take_blobs``)."""
        return self._coord_to_row

    @property
    def blob_column(self) -> str:
        """Name of the Lance blob column."""
        return self._blob_column

    @property
    def dataset(self) -> lance.LanceDataset:
        """Underlying Lance dataset (e.g. for ``take_blobs`` batching)."""
        return self._ds

    def decode_blob(self, data: bytes) -> np.ndarray:
        """Decode one stored blob to a tile (shape ``chunks``, dtype ``self.dtype``)."""
        return self._decode_tile(data)

    @property
    def n_tile_rows(self) -> int:
        return self.shape[0] // self.chunks[0]

    @property
    def n_tile_cols(self) -> int:
        return self.shape[1] // self.chunks[1]

    def __getitem__(self, key: tuple[slice, slice]) -> np.ndarray:
        r0, r1 = normalize_chunk_slices(key[0], self.shape[0])
        c0, c1 = normalize_chunk_slices(key[1], self.shape[1])
        ch0, ch1 = self.chunks
        if (r1 - r0) != ch0 or (c1 - c0) != ch1:
            raise ValueError("slice must match one full chunk shape")
        if r0 % ch0 or c0 % ch1:
            raise ValueError("slice must be chunk-aligned")
        ti, tj = r0 // ch0, c0 // ch1
        rid = self._coord_to_row[(ti, tj)]
        raw = self._ds.take_blobs(self._blob_column, indices=[rid])[0].read()
        return self._decode_tile(raw)

    def __setitem__(self, key: tuple[slice, slice], value: np.ndarray) -> None:
        raise NotImplementedError(
            "LanceBlobTileArray is read-only; use write_lance_blob_tile_dataset() "
            "to create a dataset from a full numpy raster."
        )

    def to_numpy(self) -> np.ndarray:
        """Decode every tile and assemble a full ``(H, W)`` array."""
        h, w = self.shape
        ch0, ch1 = self.chunks
        out = np.empty((h, w), dtype=self.dtype)
        for ti in range(self.n_tile_rows):
            for tj in range(self.n_tile_cols):
                r0, c0 = ti * ch0, tj * ch1
                out[r0 : r0 + ch0, c0 : c0 + ch1] = self[r0 : r0 + ch0, c0 : c0 + ch1]
        return out


def write_lance_blob_tile_dataset(
    path: str | Path,
    image: np.ndarray,
    chunk_shape: tuple[int, int],
    encode_tile: Callable[[np.ndarray], bytes],
    decode_tile: Callable[[bytes], np.ndarray],
    *,
    blob_column: str = "blob",
    data_storage_version: str = "2.2",
) -> LanceBlobTileArray:
    """Write a 2D ``image`` as one encoded blob per row and return a :class:`LanceBlobTileArray`.

    The on-disk table has columns ``row_id``, ``i``, ``j`` (tile indices), and
    ``blob`` (blob v2). ``decode_tile`` must invert ``encode_tile`` for every tile.

    Parameters
    ----------
    path
        Output dataset directory.
    image
        Full raster ``(H, W)``. ``H`` / ``W`` must be divisible by ``chunk_shape``.
    chunk_shape
        ``(ch0, ch1)`` height and width of each tile.
    encode_tile
        Tile array -> ``bytes`` written to Lance.
    decode_tile
        ``bytes`` -> tile array of shape ``chunk_shape`` (same dtype as stored).

    Returns
    -------
    LanceBlobTileArray
        Read-only view over the written dataset.
    """
    if image.ndim != 2:
        raise ValueError("image must be 2D")
    h, w = int(image.shape[0]), int(image.shape[1])
    ch0, ch1 = chunk_shape
    if h % ch0 or w % ch1:
        raise ValueError(f"shape {(h, w)} not divisible by chunk_shape {chunk_shape}")

    rows, cols = h // ch0, w // ch1
    coord_to_row: dict[tuple[int, int], int] = {}
    i_list: list[int] = []
    j_list: list[int] = []
    row_ids: list[int] = []
    blob_bytes: list[bytes] = []

    row_id = 0
    for ti in range(rows):
        for tj in range(cols):
            r0 = ti * ch0
            c0 = tj * ch1
            tile = image[r0 : r0 + ch0, c0 : c0 + ch1]
            blob_bytes.append(encode_tile(tile))
            i_list.append(ti)
            j_list.append(tj)
            row_ids.append(row_id)
            coord_to_row[(ti, tj)] = row_id
            row_id += 1

    schema = pa.schema(
        [
            pa.field("row_id", pa.int64()),
            pa.field("i", pa.int32()),
            pa.field("j", pa.int32()),
            blob_field(blob_column),
        ]
    )
    table = pa.table(
        {
            "row_id": pa.array(row_ids, type=pa.int64()),
            "i": pa.array(i_list, type=pa.int32()),
            "j": pa.array(j_list, type=pa.int32()),
            blob_column: blob_array(blob_bytes),
        },
        schema=schema,
    )
    lance.write_dataset(table, str(path), data_storage_version=data_storage_version)
    ds = lance.dataset(str(path))
    return LanceBlobTileArray(
        ds,
        chunk_shape,
        (h, w),
        coord_to_row,
        decode_tile,
        blob_column=blob_column,
        dtype=np.dtype(image.dtype),
    )
