"""
Chunk-aligned 2D tile access backed by Lance.

`LanceArray` supports NumPy-style **2D indexing**: ``int``, ``slice`` (any
step), ``...``, row-only keys (implicit ``:`` on columns), integer or boolean
``numpy`` masks, and ``list`` indices. Overlapping tiles are fetched with batched
``take_blobs`` and stitched (same idea as Zarr reading overlapping chunks).

**Lance:** each row holds one encoded tile; reads use ``Dataset.take_blobs``.
To create a dataset use `LanceArray.to_lance`. To **modify** tiles in place,
open with ``mode="r+"`` and assign with basic indices (see `LanceArray.open`).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlparse

import lance
import numpy as np
import pyarrow as pa
from lance import blob_array, blob_field

MANIFEST_FILENAME = "lance_array.json"
MANIFEST_VERSION = 1

_REMOTE_URI_SCHEMES = frozenset({"s3", "gs", "gcs", "https", "http"})


def _is_remote_dataset_uri(path_str: str) -> bool:
    return urlparse(path_str).scheme.lower() in _REMOTE_URI_SCHEMES


def _load_manifest_json(path: str | Path) -> tuple[dict[str, Any], str]:
    """Return ``(manifest_dict, uri_for_lance_dataset)``."""
    path_str = str(path).strip()
    if _is_remote_dataset_uri(path_str):
        try:
            import smart_open
        except ImportError as e:
            raise ImportError(
                "Opening remote dataset URIs (s3://, gs://, https://, …) requires "
                "smart-open. Install with: pip install 'lance-array[cloud]' "
                "or pip install smart-open"
            ) from e
        base = path_str.rstrip("/")
        man_url = f"{base}/{MANIFEST_FILENAME}"
        with smart_open.open(man_url, "r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
        return data, base
    root = Path(path).expanduser().resolve()
    man_path = root / MANIFEST_FILENAME
    if not man_path.is_file():
        raise FileNotFoundError(
            f"missing {MANIFEST_FILENAME} under {root}; "
            "open() expects a dataset created with LanceArray.to_lance()."
        )
    data = json.loads(man_path.read_text(encoding="utf-8"))
    return data, str(root)


__all__ = [
    "LanceArray",
    "TileCodec",
    "normalize_chunk_slices",
    "open_array",
]


class TileCodec(Enum):
    """How each tile is encoded in the Lance blob column.

    Pass a member (or string alias such as ``\"raw\"``, ``\"blosc_numcodecs\"``,
    ``\"blosc2\"``) to `LanceArray.to_lance`. ``BLOSC2`` requires the ``blosc2``
    package (e.g. ``lance-array[zarr]``). See enum members below for each preset.
    """

    RAW = "raw"
    """Uncompressed contiguous bytes (dtype itemsize × cells per tile)."""

    BLOSC_NUMCODECS = "blosc_numcodecs"
    """Blosc1 via ``numcodecs.Blosc`` (typical Zarr Blosc parity)."""

    BLOSC2 = "blosc2"
    """Blosc2 via ``blosc2.compress`` / ``decompress`` (install ``blosc2``)."""


def _coerce_tile_codec(codec: TileCodec | str) -> TileCodec:
    if isinstance(codec, TileCodec):
        return codec
    key = str(codec).strip().lower().replace("-", "_")
    for c in TileCodec:
        if c.value == key:
            return c
    opts = ", ".join(c.value for c in TileCodec)
    raise ValueError(f"unknown tile codec {codec!r}; expected one of: {opts}")


def _blosc2_codec_id(cname: str):
    import blosc2

    name = cname.strip().lower()
    if name == "zstd":
        return blosc2.Codec.ZSTD
    if name == "lz4":
        return blosc2.Codec.LZ4
    if name in ("blosclz", "blosc"):
        return blosc2.Codec.BLOSCLZ
    raise ValueError(
        f"unsupported blosc2 codec name {cname!r} for TileCodec.BLOSC2 (try 'zstd')"
    )


def _build_tile_codecs(
    chunk_shape: tuple[int, int],
    dtype: np.dtype,
    codec: TileCodec,
    *,
    blosc_typesize: int | None,
    blosc_clevel: int,
    blosc_cname: str,
) -> tuple[Callable[[np.ndarray], bytes], Callable[[bytes], np.ndarray]]:
    ch0, ch1 = chunk_shape
    dtype = np.dtype(dtype)
    itemsize = int(dtype.itemsize)
    typesize = int(blosc_typesize) if blosc_typesize is not None else itemsize
    nbytes = ch0 * ch1 * itemsize

    if codec is TileCodec.RAW:

        def encode_tile(arr: np.ndarray) -> bytes:
            data = np.ascontiguousarray(arr, dtype=dtype)
            return data.tobytes()

        def decode_tile(blob: bytes) -> np.ndarray:
            if len(blob) != nbytes:
                raise ValueError(f"raw tile expected {nbytes} bytes, got {len(blob)}")
            return np.frombuffer(blob, dtype=dtype).reshape(chunk_shape)

        return encode_tile, decode_tile

    if codec is TileCodec.BLOSC_NUMCODECS:
        try:
            from numcodecs import Blosc
        except ImportError as e:
            raise ImportError(
                "TileCodec.BLOSC_NUMCODECS requires numcodecs (core dependency of lance-array)."
            ) from e

        blosc = Blosc(
            cname=blosc_cname,
            clevel=blosc_clevel,
            shuffle=Blosc.SHUFFLE,
            typesize=typesize,
            blocksize=0,
        )

        def encode_tile(arr: np.ndarray) -> bytes:
            data = np.ascontiguousarray(arr, dtype=dtype)
            return blosc.encode(data.tobytes())

        def decode_tile(blob: bytes) -> np.ndarray:
            raw = blosc.decode(blob)
            if len(raw) != nbytes:
                raise ValueError(
                    f"decompressed tile expected {nbytes} bytes, got {len(raw)}"
                )
            return np.frombuffer(raw, dtype=dtype).reshape(chunk_shape)

        return encode_tile, decode_tile

    if codec is TileCodec.BLOSC2:
        try:
            import blosc2
        except ImportError as e:
            raise ImportError(
                "TileCodec.BLOSC2 requires blosc2. "
                "Install with `pip install blosc2` or the package extra that "
                "includes it (e.g. `lance-array[zarr]`)."
            ) from e

        codec_id = _blosc2_codec_id(blosc_cname)

        def encode_tile(arr: np.ndarray) -> bytes:
            data = np.ascontiguousarray(arr, dtype=dtype)
            # blosc2 defaults typesize to 8 for plain buffers; always pass the dtype
            # itemsize (e.g. 2 for uint16) so shuffle matches numcodecs / Zarr Blosc.
            buf = data.tobytes()
            return cast(
                bytes,
                blosc2.compress(
                    buf,
                    typesize=typesize,
                    clevel=blosc_clevel,
                    filter=blosc2.Filter.SHUFFLE,
                    codec=codec_id,
                ),
            )

        def decode_tile(blob: bytes) -> np.ndarray:
            raw = blosc2.decompress(blob)
            if not isinstance(raw, bytes | bytearray | memoryview):
                raise TypeError(
                    f"blosc2.decompress expected buffer, got {type(raw).__name__}"
                )
            raw = bytes(raw)
            if len(raw) != nbytes:
                raise ValueError(
                    f"decompressed tile expected {nbytes} bytes, got {len(raw)}"
                )
            return np.frombuffer(raw, dtype=dtype).reshape(chunk_shape)

        return encode_tile, decode_tile

    raise AssertionError(f"unhandled codec {codec!r}")


def normalize_chunk_slices(s: slice, dim: int) -> tuple[int, int]:
    """Normalize a slice to ``(start, stop)`` with step ``1`` and positive span.

    Parameters
    ----------
    s
        Slice along an axis of logical length ``dim`` (uses ``s.indices(dim)``).
    dim
        Size of that axis.

    Returns
    -------
    tuple[int, int]
        Half-open interval ``(start, stop)``.

    Raises
    ------
    ValueError
        If ``step != 1`` or the slice is empty after normalization.
    """
    start, stop, step = s.indices(dim)
    if step != 1:
        raise ValueError("step must be 1")
    if stop <= start:
        raise ValueError("empty slice not supported")
    return start, stop


_TAKE_BLOBS_BATCH = 512


def _write_lance_manifest(
    root: Path,
    *,
    shape: tuple[int, int],
    chunk_shape: tuple[int, int],
    dtype: np.dtype,
    blob_column: str,
    codec: TileCodec,
    blosc_typesize: int | None,
    blosc_clevel: int,
    blosc_cname: str,
) -> None:
    dt = np.dtype(dtype)
    payload: dict[str, Any] = {
        "version": MANIFEST_VERSION,
        "format": "lance_array",
        "shape": [int(shape[0]), int(shape[1])],
        "chunk_shape": [int(chunk_shape[0]), int(chunk_shape[1])],
        "dtype": dt.str,
        "blob_column": blob_column,
        "codec": codec.value,
    }
    if codec in (TileCodec.BLOSC_NUMCODECS, TileCodec.BLOSC2):
        payload["blosc_typesize"] = int(
            blosc_typesize if blosc_typesize is not None else dt.itemsize
        )
        payload["blosc_clevel"] = int(blosc_clevel)
        payload["blosc_cname"] = str(blosc_cname)
    (root / MANIFEST_FILENAME).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _load_coord_mapping(ds: lance.LanceDataset) -> dict[tuple[int, int], int]:
    """Map tile ``(i, j)`` to **positional** row index for ``take_blobs``.

    Positions follow ``to_table(columns=[\"i\", \"j\"])`` row order. This must not
    use the ``row_id`` column: after merge-insert, physical order can differ from
    stored ``row_id`` while ``take_blobs(indices=…)`` is positional.
    """
    tbl = ds.to_table(columns=["i", "j"])
    coord_to_row: dict[tuple[int, int], int] = {}
    for pos in range(tbl.num_rows):
        ti = int(tbl["i"][pos].as_py())
        tj = int(tbl["j"][pos].as_py())
        coord_to_row[(ti, tj)] = pos
    return coord_to_row


def open_array(store: str | Path, *, mode: str = "r") -> LanceArray:
    """Open a Lance tile dataset (Zarr-style entry point).

    Like ``zarr.open_array``, but for a dataset written by `LanceArray.to_lance`
    (includes ``lance_array.json``).

    Parameters
    ----------
    store
        Dataset directory or URI passed to ``lance.dataset``.
    mode
        ``"r"`` read-only. ``"r+"`` allows `LanceArray.__setitem__` for basic
        indices (see PRD ``prds/lance-array-slice-writes.md``).

    Returns
    -------
    LanceArray
        Same as `LanceArray.open`.

    Raises
    ------
    ValueError
        Invalid ``mode``, corrupt manifest, or dataset row count mismatch.
    FileNotFoundError
        Missing ``lance_array.json`` for a local path.
    ImportError
        Remote URI without ``smart-open`` installed.

    Notes
    -----
    For ``s3://``, ``gs://``, or ``https://``, install ``smart-open`` (extra
    ``lance-array[cloud]``). The manifest is read with ``smart_open`` before
    opening Lance.
    """
    return LanceArray.open(store, mode=mode)


def _pack_2d_index_result(
    out: np.ndarray, row_was_int: bool, col_was_int: bool
) -> np.ndarray:
    """Match NumPy rank rules for (int|slice, int|slice) indexing."""
    if row_was_int and col_was_int:
        return np.array(out[0, 0], dtype=out.dtype)
    if row_was_int:
        return np.ascontiguousarray(out[0, :])
    if col_was_int:
        return np.ascontiguousarray(out[:, 0])
    return out


def _normalize_lance_key(key: Any) -> tuple[Any, Any]:
    """Turn user ``__getitem__`` keys into ``(row_spec, col_spec)`` (NumPy 2D rules)."""
    if key is Ellipsis:
        return (slice(None), slice(None))
    if not isinstance(key, tuple):
        return (key, slice(None))
    if len(key) == 0:
        raise IndexError("an index tuple must have at least one entry")
    if len(key) == 1:
        k0 = key[0]
        if k0 is Ellipsis:
            return (slice(None), slice(None))
        return (k0, slice(None))
    if sum(k is Ellipsis for k in key) > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    if any(k is Ellipsis for k in key):
        i = key.index(Ellipsis)
        before = list(key[:i])
        after = list(key[i + 1 :])
        n_fill = 2 - len(before) - len(after)
        if n_fill < 0:
            raise IndexError("too many indices for 2D array")
        full = before + [slice(None)] * n_fill + after
        if len(full) != 2:
            raise IndexError("too many indices for 2D array")
        return (full[0], full[1])
    if len(key) == 2:
        return (key[0], key[1])
    raise IndexError("too many indices for 2D array")


def _coerce_1d_bool_index(spec: Any) -> np.ndarray | None:
    if isinstance(spec, list):
        a = np.asarray(spec)
    elif isinstance(spec, np.ndarray):
        a = spec
    else:
        return None
    if a.ndim == 1 and (a.dtype == bool or a.dtype == np.dtype("?")):
        return a
    return None


def _reject_python_bool_scalar(spec: Any, *, kind: str) -> None:
    if isinstance(spec, bool):
        raise TypeError(
            f"{kind} index must be int, slice, or a numpy array; "
            "use a numpy boolean ndarray for masking"
        )


def _classify_axis(spec: Any, dim: int, *, kind: str) -> tuple[Any, ...]:
    """Return ``('scalar', i)``, ``('slice', start, stop, step)``, or ``('adv', idx)``."""
    _reject_python_bool_scalar(spec, kind=kind)
    if isinstance(spec, int | np.integer):
        i = int(spec) + dim if int(spec) < 0 else int(spec)
        if not 0 <= i < dim:
            raise IndexError(
                f"index {spec} is out of bounds for axis size {dim} ({kind} axis)"
            )
        return ("scalar", i)
    if isinstance(spec, slice):
        s, e, st = spec.indices(dim)
        return ("slice", s, e, st)
    if isinstance(spec, list):
        return _classify_axis(np.asarray(spec), dim, kind=kind)
    if isinstance(spec, np.ndarray):
        if spec.dtype == bool or spec.dtype == np.dtype("?"):
            if spec.ndim != 1 or int(spec.shape[0]) != dim:
                raise IndexError(
                    f"boolean {kind} index has length {spec.shape[0]}; "
                    f"axis length is {dim}"
                )
            nz = np.flatnonzero(spec).astype(np.intp, copy=False)
            return ("adv", nz)
        if spec.ndim == 0:
            return _classify_axis(int(spec.item()), dim, kind=kind)
        if not np.issubdtype(spec.dtype, np.integer):
            raise TypeError(
                f"unsupported {kind} numpy index dtype {spec.dtype!r} "
                "(use integer or boolean dtype)"
            )
        t = np.asarray(spec, dtype=np.intp, order="C")
        flat = t.ravel()
        if flat.size == 0:
            return ("adv", t)
        neg = flat < 0
        adj = flat.copy()
        adj[neg] += dim
        if adj.min() < 0 or adj.max() >= dim:
            raise IndexError(f"out-of-bounds {kind} index for axis size {dim}")
        return ("adv", t)
    raise TypeError(
        f"unsupported {kind} index type {type(spec).__name__!r} "
        "(use int, slice, list, or numpy array)"
    )


def _slice_slice_strided(
    rs: int,
    re: int,
    rst: int,
    cs: int,
    ce: int,
    cst: int,
    read_subarray: Callable[..., np.ndarray],
    dtype: np.dtype,
) -> np.ndarray:
    ri = np.arange(rs, re, rst, dtype=np.intp)
    ci = np.arange(cs, ce, cst, dtype=np.intp)
    if ri.size == 0 or ci.size == 0:
        return np.empty((ri.size, ci.size), dtype=dtype)
    r0, r1 = int(ri.min()), int(ri.max()) + 1
    c0, c1 = int(ci.min()), int(ci.max()) + 1
    sub = read_subarray(r0, r1, c0, c1)
    return np.ascontiguousarray(sub[np.ix_(ri - r0, ci - c0)])


def _gather_at_pairs(
    Ri: np.ndarray,
    Ci: np.ndarray,
    read_subarray: Callable[..., np.ndarray],
    dtype: np.dtype,
) -> np.ndarray:
    if Ri.shape != Ci.shape:
        raise AssertionError("internal: Ri/Ci shape mismatch")
    if Ri.size == 0:
        return np.empty(Ri.shape, dtype=dtype)
    r0, r1 = int(Ri.min()), int(Ri.max()) + 1
    c0, c1 = int(Ci.min()), int(Ci.max()) + 1
    sub = read_subarray(r0, r1, c0, c1)
    return np.ascontiguousarray(sub[Ri - r0, Ci - c0])


def _gather_two_bool_masks(
    mr: np.ndarray,
    mc: np.ndarray,
    read_subarray: Callable[..., np.ndarray],
    dtype: np.dtype,
) -> np.ndarray:
    """NumPy semantics for ``array[mask_row, mask_col]`` with two 1-D boolean masks."""
    r = np.flatnonzero(mr).astype(np.intp, copy=False)
    c = np.flatnonzero(mc).astype(np.intp, copy=False)
    if r.size == c.size:
        return _gather_at_pairs(r, c, read_subarray, dtype)
    try:
        br, bc = np.broadcast_arrays(r, c)
    except ValueError as e:
        raise IndexError(
            "shape mismatch: boolean masks' nonzero indices could not be "
            "broadcast together"
        ) from e
    return _gather_at_pairs(br, bc, read_subarray, dtype)


def _combine_for_advanced(
    r_cls: tuple[Any, ...], c_cls: tuple[Any, ...], _h: int, _w: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build broadcast row/col index arrays (NumPy advanced + basic rules for 2D)."""
    if r_cls[0] != "adv" and c_cls[0] != "adv":
        raise AssertionError("expected at least one advanced axis")

    if r_cls[0] == "adv" and c_cls[0] == "adv":
        br, bc = np.broadcast_arrays(r_cls[1], c_cls[1])
        return br, bc

    if r_cls[0] == "adv":
        ri = np.asarray(r_cls[1], dtype=np.intp, order="C")
        if c_cls[0] == "scalar":
            ci = np.full(ri.shape, c_cls[1], dtype=np.intp)
            return ri, ci
        if c_cls[0] == "slice":
            cs, ce, cst = c_cls[1], c_cls[2], c_cls[3]
            ci_1d = np.arange(cs, ce, cst, dtype=np.intp)
            ell = ci_1d.size
            ri_b = np.broadcast_to(ri[..., np.newaxis], ri.shape + (ell,))
            ci_b = np.broadcast_to(
                ci_1d.reshape((1,) * ri.ndim + (ell,)),
                ri_b.shape,
            )
            return ri_b, ci_b
        raise AssertionError("unreachable")

    # c_cls[0] == "adv"
    ci = np.asarray(c_cls[1], dtype=np.intp, order="C")
    if r_cls[0] == "scalar":
        ri = np.full(ci.shape, r_cls[1], dtype=np.intp)
        return ri, ci
    if r_cls[0] == "slice":
        # NumPy order: slice axis first, then advanced (e.g. ``a[4:40, [1,2,3]]`` → (36, 3)).
        rs, re, rst = r_cls[1], r_cls[2], r_cls[3]
        ri_1d = np.arange(rs, re, rst, dtype=np.intp)
        ell = ri_1d.size
        lead = (ell,) + tuple(1 for _ in range(ci.ndim))
        Ri = np.broadcast_to(ri_1d.reshape(lead), (ell,) + ci.shape)
        Ci = np.broadcast_to(ci.reshape((1,) + ci.shape), (ell,) + ci.shape)
        return Ri, Ci
    raise AssertionError("unreachable")


def _stored_row_id_by_tile(ds: lance.LanceDataset) -> dict[tuple[int, int], int]:
    """``(i, j)`` → ``row_id`` column value (for merge-insert source rows)."""
    snap = ds.to_table(columns=["row_id", "i", "j"])
    out: dict[tuple[int, int], int] = {}
    for row in range(snap.num_rows):
        ij = (int(snap["i"][row].as_py()), int(snap["j"][row].as_py()))
        out[ij] = int(snap["row_id"][row].as_py())
    return out


def _basic_setitem_meshes(
    r_cls: tuple[Any, ...], c_cls: tuple[Any, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Row/col index meshes for NumPy basic assignment (scalar or slice per axis, step 1)."""
    step_err = NotImplementedError(
        "slice assignment with step != 1 is not implemented; use a contiguous slice"
    )
    if r_cls[0] == "scalar" and c_cls[0] == "scalar":
        r, c = int(r_cls[1]), int(c_cls[1])
        return (
            np.array([[r]], dtype=np.intp),
            np.array([[c]], dtype=np.intp),
        )
    if r_cls[0] == "scalar" and c_cls[0] == "slice":
        rs, re, st = c_cls[1], c_cls[2], c_cls[3]
        if st != 1:
            raise step_err
        r = int(r_cls[1])
        ci = np.arange(rs, re, st, dtype=np.intp)
        Ri = np.full((1, ci.size), r, dtype=np.intp)
        Ci = ci.reshape(1, -1)
        return Ri, Ci
    if r_cls[0] == "slice" and c_cls[0] == "scalar":
        rs, re, st = r_cls[1], r_cls[2], r_cls[3]
        if st != 1:
            raise step_err
        c = int(c_cls[1])
        ri = np.arange(rs, re, st, dtype=np.intp)
        Ri = ri.reshape(-1, 1)
        Ci = np.full((ri.size, 1), c, dtype=np.intp)
        return Ri, Ci
    if r_cls[0] == "slice" and c_cls[0] == "slice":
        rs, re, rst = r_cls[1], r_cls[2], r_cls[3]
        cs, ce, cst = c_cls[1], c_cls[2], c_cls[3]
        if rst != 1 or cst != 1:
            raise step_err
        ri = np.arange(rs, re, rst, dtype=np.intp)
        ci = np.arange(cs, ce, cst, dtype=np.intp)
        Ri, Ci = np.meshgrid(ri, ci, indexing="ij")
        return Ri, Ci
    raise AssertionError("internal: expected basic row and column specs")


class LanceArray:
    """2D view over a Lance dataset with one encoded tile per row.

    Rows are indexed by logical tile grid ``(tile_i, tile_j)`` mapped to a Lance
    **positional** row index for ``take_blobs``. Each stored payload is decoded
    using the `TileCodec` chosen at write time; see `decode_tile`.

    **Indexing:** NumPy-like ``view[row, col]`` — ``int``, ``slice`` (including
    step ≠ 1), ``...``, ``view[row]`` as ``view[row, :]``, and advanced indices
    (integer or boolean ``ndarray``, ``list``) with the same broadcasting rules
    as NumPy for 2D arrays. Overlapping tiles are read via batched ``take_blobs``
    and stitched (including partial edge tiles).

    **Assignment:** only for views opened with ``mode="r+"``. Supported keys match
    basic NumPy indexing with **slice step 1** on both axes; fancy and boolean
    assignment is not implemented.

    **Full raster:** `to_numpy` materializes the entire grid in one batched read
    path.

    **Create on disk:** `LanceArray.to_lance` writes a new dataset from a NumPy
    image and returns a read-only view; open with ``mode="r+"`` to mutate.

    Attributes
    ----------
    shape : tuple[int, int]
        Raster shape ``(H, W)``.
    chunks : tuple[int, int]
        Tile shape ``(ch0, ch1)``.
    dtype : numpy.dtype
        Pixel dtype of the logical raster.
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
        encode_tile: Callable[[np.ndarray], bytes] | None = None,
    ) -> None:
        """Build a view from an open Lance dataset (prefer `LanceArray.open` or `LanceArray.to_lance`).

        Parameters
        ----------
        dataset
            Open ``lance.LanceDataset`` with one row per tile and ``(i, j)`` keys.
        chunk_shape
            ``(ch0, ch1)`` tile size in pixels.
        image_shape
            Full raster shape ``(H, W)``.
        coord_to_row
            Map ``(tile_i, tile_j)`` to the **positional** row index used by
            ``take_blobs`` (see module helpers).
        decode_tile
            Decode one blob payload to a ``chunk_shape`` array.
        blob_column
            Lance blob column name for tile payloads.
        dtype
            Raster dtype; defaults to ``uint16`` if omitted.
        encode_tile
            Encode a tile array to bytes; must be set for ``r+`` writes, else
            ``None`` for read-only views.
        """
        self._ds = dataset
        self.shape: tuple[int, int] = image_shape
        self.chunks: tuple[int, int] = chunk_shape
        self.dtype: np.dtype = (
            np.dtype(dtype) if dtype is not None else np.dtype(np.uint16)
        )
        self._coord_to_row = coord_to_row
        self._decode_tile = decode_tile
        self._encode_tile = encode_tile
        self._blob_column = blob_column

    @property
    def ndim(self) -> int:
        """Number of dimensions.

        Returns
        -------
        int
            Always ``2``.
        """
        return 2

    @classmethod
    def open(cls, path: str | Path, *, mode: str = "r") -> LanceArray:
        """Open a dataset written with `LanceArray.to_lance` using a sidecar manifest.

        The dataset root must expose ``lance_array.json`` (written by `LanceArray.to_lance`).
        For local paths this is a file under the dataset directory; for URIs (e.g.
        ``s3://...``) the manifest is read via optional ``smart-open``.

        Parameters
        ----------
        path
            Lance dataset directory or URI (e.g. ``s3://bucket/prefix/array.lance``).
            Remote roots need the optional ``smart-open`` dependency
            (``pip install 'lance-array[cloud]'``); credentials follow the normal
            cloud SDK / environment defaults.
        mode
            ``"r"`` read-only. ``"r+"`` allows `__setitem__` for basic indices
            (contiguous slices and integers; slice step must be ``1``).

        Returns
        -------
        LanceArray
            View over the on-disk dataset; use ``mode="r+"`` for slice assignment.

        Raises
        ------
        ValueError
            If ``mode`` is invalid, the manifest is missing/unsupported, or the
            table does not match manifest shape/chunks.
        FileNotFoundError
            If ``lance_array.json`` is missing for a local path.
        ImportError
            If a remote URI is used without ``smart-open`` installed.
        """
        if mode not in ("r", "r+"):
            raise ValueError(f"invalid mode {mode!r}")
        data, dataset_uri = _load_manifest_json(path)
        if (
            data.get("format") != "lance_array"
            or data.get("version") != MANIFEST_VERSION
        ):
            raise ValueError(
                f"unsupported or corrupt manifest for Lance dataset {dataset_uri!r}"
            )
        shape = (int(data["shape"][0]), int(data["shape"][1]))
        chunk_shape = (int(data["chunk_shape"][0]), int(data["chunk_shape"][1]))
        dtype = np.dtype(data["dtype"])
        blob_column = str(data["blob_column"])
        codec = _coerce_tile_codec(data["codec"])
        blosc_typesize = data.get("blosc_typesize")
        if blosc_typesize is not None:
            blosc_typesize = int(blosc_typesize)
        blosc_clevel = int(data.get("blosc_clevel", 5))
        blosc_cname = str(data.get("blosc_cname", "zstd"))
        enc, dec = _build_tile_codecs(
            chunk_shape,
            dtype,
            codec,
            blosc_typesize=blosc_typesize,
            blosc_clevel=blosc_clevel,
            blosc_cname=blosc_cname,
        )
        ds = lance.dataset(dataset_uri)
        coord_to_row = _load_coord_mapping(ds)
        ch0, ch1 = chunk_shape
        expected_tiles = (shape[0] // ch0) * (shape[1] // ch1)
        if len(coord_to_row) != expected_tiles:
            raise ValueError(
                f"manifest shape/chunks imply {expected_tiles} tiles but dataset has "
                f"{len(coord_to_row)} rows with (i,j) keys"
            )
        encode_tile = enc if mode == "r+" else None
        return cls(
            ds,
            chunk_shape,
            shape,
            coord_to_row,
            dec,
            blob_column=blob_column,
            dtype=dtype,
            encode_tile=encode_tile,
        )

    @classmethod
    def to_lance(
        cls,
        path: str | Path,
        image: np.ndarray,
        chunk_shape: tuple[int, int],
        *,
        codec: TileCodec | str = TileCodec.RAW,
        blosc_typesize: int | None = None,
        blosc_clevel: int = 5,
        blosc_cname: str = "zstd",
        blob_column: str = "blob",
        data_storage_version: Literal[
            "stable", "2.0", "2.1", "2.2", "2.3", "next", "legacy", "0.1"
        ] = "2.2",
    ) -> LanceArray:
        """Write a 2D ``image`` as one encoded tile per row and return a `LanceArray`.

        The on-disk table has columns ``row_id``, ``i``, ``j`` (tile indices), and
        a blob column (default name ``blob``, blob v2). A sidecar ``lance_array.json``
        stores shape, chunk grid, dtype, and codec parameters so `LanceArray.open` works.

        Pass ``codec=`` as `TileCodec` or a string alias (``\"raw\"``,
        ``\"blosc_numcodecs\"``, ``\"blosc2\"``). Blosc presets use ``blosc_typesize``
        (defaults to ``dtype`` itemsize), ``blosc_clevel``, and ``blosc_cname`` where
        applicable.

        Parameters
        ----------
        path
            Output dataset directory.
        image
            Full raster ``(H, W)``. ``H`` / ``W`` must be divisible by ``chunk_shape``.
        chunk_shape
            ``(ch0, ch1)`` height and width of each tile.
        codec
            Built-in tile codec preset.
        blosc_typesize
            Blosc ``typesize`` for ``BLOSC_NUMCODECS`` / ``BLOSC2`` (default: dtype itemsize).
        blosc_clevel
            Blosc compression level.
        blosc_cname
            Blosc compressor name (e.g. ``\"zstd\"``). For ``BLOSC2``, only a subset
            is mapped (``zstd``, ``lz4``, ``blosclz``).
        blob_column
            Name of the blob column in the Lance schema.
        data_storage_version
            Passed to ``lance.write_dataset``.

        Returns
        -------
        LanceArray
            Read-only view over the written dataset (use `LanceArray.open` with
            ``mode="r+"`` to assign slices).

        Raises
        ------
        ValueError
            If ``image`` is not 2D, shape is not divisible by ``chunk_shape``, or
            codec options are invalid.
        """
        enc, dec = _build_tile_codecs(
            chunk_shape,
            np.dtype(image.dtype),
            _coerce_tile_codec(codec),
            blosc_typesize=blosc_typesize,
            blosc_clevel=blosc_clevel,
            blosc_cname=blosc_cname,
        )

        if image.ndim != 2:
            raise ValueError("image must be 2D")
        h, w = int(image.shape[0]), int(image.shape[1])
        ch0, ch1 = chunk_shape
        if h % ch0 or w % ch1:
            raise ValueError(
                f"shape {(h, w)} not divisible by chunk_shape {chunk_shape}"
            )

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
                blob_bytes.append(enc(tile))
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
        root = Path(path)
        tc = _coerce_tile_codec(codec)
        _write_lance_manifest(
            root,
            shape=(h, w),
            chunk_shape=(ch0, ch1),
            dtype=np.dtype(image.dtype),
            blob_column=blob_column,
            codec=tc,
            blosc_typesize=blosc_typesize,
            blosc_clevel=blosc_clevel,
            blosc_cname=blosc_cname,
        )
        ds = lance.dataset(str(path))
        return cls(
            ds,
            chunk_shape,
            (h, w),
            coord_to_row,
            dec,
            blob_column=blob_column,
            dtype=np.dtype(image.dtype),
            encode_tile=None,
        )

    @property
    def coord_to_row(self) -> dict[tuple[int, int], int]:
        """Map each tile grid index to its Lance **positional** row index.

        Returns
        -------
        dict[tuple[int, int], int]
            Keys ``(tile_i, tile_j)``; values are indices for ``take_blobs``.
        """
        return self._coord_to_row

    @property
    def blob_column(self) -> str:
        """Lance column name holding encoded tile payloads.

        Returns
        -------
        str
            Blob v2 column name (default ``\"blob\"`` unless overridden at write).
        """
        return self._blob_column

    @property
    def dataset(self) -> lance.LanceDataset:
        """Underlying Lance dataset handle.

        Returns
        -------
        lance.LanceDataset
            Use e.g. ``take_blobs`` for advanced access; most users rely on
            `__getitem__` / `to_numpy` instead.
        """
        return self._ds

    def decode_tile(self, data: bytes) -> np.ndarray:
        """Decode one blob from storage into a single tile array.

        Parameters
        ----------
        data
            Raw bytes from the blob column for one row.

        Returns
        -------
        numpy.ndarray
            Shape ``chunks``, dtype ``self.dtype``.
        """
        return self._decode_tile(data)

    @property
    def n_tile_rows(self) -> int:
        """Number of tile rows along axis 0.

        Returns
        -------
        int
            ``shape[0] // chunks[0]``.
        """
        return self.shape[0] // self.chunks[0]

    @property
    def n_tile_cols(self) -> int:
        """Number of tile columns along axis 1.

        Returns
        -------
        int
            ``shape[1] // chunks[1]``.
        """
        return self.shape[1] // self.chunks[1]

    def _getitem_both_basic(
        self, r_cls: tuple[Any, ...], c_cls: tuple[Any, ...]
    ) -> np.ndarray:
        """Fast paths when both axes are only ``int`` or ``slice`` (no fancy indices)."""
        if r_cls[0] not in ("scalar", "slice") or c_cls[0] not in ("scalar", "slice"):
            raise AssertionError("internal: expected basic row and column specs")

        if r_cls[0] == "scalar" and c_cls[0] == "scalar":
            sub = self._read_subarray(r_cls[1], r_cls[1] + 1, c_cls[1], c_cls[1] + 1)
            return np.array(sub[0, 0], dtype=sub.dtype)

        if r_cls[0] == "scalar" and c_cls[0] == "slice":
            i = r_cls[1]
            s, e, st = c_cls[1], c_cls[2], c_cls[3]
            if st == 1:
                sub = self._read_subarray(i, i + 1, s, e)
                return _pack_2d_index_result(sub, True, False)
            ri = np.array([i], dtype=np.intp)
            ci = np.arange(s, e, st, dtype=np.intp)
            if ci.size == 0:
                return np.empty((0,), dtype=self.dtype)
            Ri, Ci = np.broadcast_arrays(ri[:, np.newaxis], ci[np.newaxis, :])
            out = _gather_at_pairs(Ri, Ci, self._read_subarray, self.dtype)
            return np.ascontiguousarray(out[0])

        if r_cls[0] == "slice" and c_cls[0] == "scalar":
            s, e, st = r_cls[1], r_cls[2], r_cls[3]
            j = c_cls[1]
            if st == 1:
                sub = self._read_subarray(s, e, j, j + 1)
                return _pack_2d_index_result(sub, False, True)
            ri = np.arange(s, e, st, dtype=np.intp)
            ci = np.array([j], dtype=np.intp)
            if ri.size == 0:
                return np.empty((0,), dtype=self.dtype)
            Ri, Ci = np.broadcast_arrays(ri[:, np.newaxis], ci[np.newaxis, :])
            out = _gather_at_pairs(Ri, Ci, self._read_subarray, self.dtype)
            return np.ascontiguousarray(out[:, 0])

        rs, re, rst = r_cls[1], r_cls[2], r_cls[3]
        cs, ce, cst = c_cls[1], c_cls[2], c_cls[3]
        if rst == 1 and cst == 1:
            return self._read_subarray(rs, re, cs, ce)
        return _slice_slice_strided(
            rs, re, rst, cs, ce, cst, self._read_subarray, self.dtype
        )

    def _read_subarray(self, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
        """Read half-open rectangle ``[r0, r1) × [c0, c1)`` (may be empty)."""
        out = np.empty((r1 - r0, c1 - c0), dtype=self.dtype)
        if r0 >= r1 or c0 >= c1:
            return out

        ch0, ch1 = self.chunks
        i_start, i_end = r0 // ch0, (r1 - 1) // ch0
        j_start, j_end = c0 // ch1, (c1 - 1) // ch1

        rids_ordered: list[int] = []
        for ti in range(i_start, i_end + 1):
            for tj in range(j_start, j_end + 1):
                rids_ordered.append(self._coord_to_row[(ti, tj)])

        unique_sorted = sorted(set(rids_ordered))
        blob_cache: dict[int, np.ndarray] = {}
        col = self._blob_column
        for k in range(0, len(unique_sorted), _TAKE_BLOBS_BATCH):
            batch = unique_sorted[k : k + _TAKE_BLOBS_BATCH]
            files = self._ds.take_blobs(col, indices=batch)
            for rid, f in zip(batch, files, strict=True):
                blob_cache[rid] = self._decode_tile(f.read())

        for ti in range(i_start, i_end + 1):
            for tj in range(j_start, j_end + 1):
                rid = self._coord_to_row[(ti, tj)]
                tile = blob_cache[rid]
                R0, C0 = ti * ch0, tj * ch1
                R1, C1 = R0 + ch0, C0 + ch1
                tr0, tr1 = max(r0, R0), min(r1, R1)
                tc0, tc1 = max(c0, C0), min(c1, C1)
                if tr0 >= tr1 or tc0 >= tc1:
                    continue
                sr0, sr1 = tr0 - R0, tr1 - R0
                sc0, sc1 = tc0 - C0, tc1 - C0
                dr0, dr1 = tr0 - r0, tr1 - r0
                dc0, dc1 = tc0 - c0, tc1 - c0
                out[dr0:dr1, dc0:dc1] = tile[sr0:sr1, sc0:sc1]

        return out

    def __getitem__(self, key: Any) -> np.ndarray:
        """Read a scalar, slice, or advanced subregion (NumPy 2D semantics).

        Overlapping tiles are fetched via batched ``take_blobs``, decoded, and
        stitched (including strided slices and partial edge windows).

        Parameters
        ----------
        key
            ``int``, ``slice``, ``Ellipsis``, row-only key, integer/boolean
            ``numpy.ndarray``, or ``list`` indices; same rank and broadcasting
            rules as NumPy for a 2D array.

        Returns
        -------
        numpy.ndarray
            0-d for scalar indices, otherwise the selected subarray.

        Raises
        ------
        IndexError
            If indices are out of bounds or boolean masks do not match ``shape``.
        TypeError
            For invalid key types (e.g. Python ``bool`` instead of a boolean array).
        """
        r_spec, c_spec = _normalize_lance_key(key)
        h, w = self.shape
        br = _coerce_1d_bool_index(r_spec)
        bc = _coerce_1d_bool_index(c_spec)
        if br is not None and bc is not None:
            if int(br.shape[0]) != h or int(bc.shape[0]) != w:
                raise IndexError(
                    f"boolean row and column masks must match array shape ({h}, {w})"
                )
            return _gather_two_bool_masks(br, bc, self._read_subarray, self.dtype)

        r_cls = _classify_axis(r_spec, h, kind="row")
        c_cls = _classify_axis(c_spec, w, kind="column")

        if r_cls[0] != "adv" and c_cls[0] != "adv":
            return self._getitem_both_basic(r_cls, c_cls)

        ri, ci = _combine_for_advanced(r_cls, c_cls, h, w)
        return _gather_at_pairs(ri, ci, self._read_subarray, self.dtype)

    def _merge_update_tiles(
        self,
        tile_arrays: dict[tuple[int, int], np.ndarray],
        *,
        ij_to_stored_row_id: dict[tuple[int, int], int],
    ) -> None:
        """Persist encoded tiles via Lance merge-insert on ``(i, j)``."""
        if not tile_arrays:
            return
        col = self._blob_column
        assert self._encode_tile is not None
        i_list: list[int] = []
        j_list: list[int] = []
        rid_list: list[int] = []
        blob_bytes: list[bytes] = []
        for (ti, tj), arr in sorted(tile_arrays.items()):
            i_list.append(ti)
            j_list.append(tj)
            rid_list.append(ij_to_stored_row_id[(ti, tj)])
            blob_bytes.append(self._encode_tile(arr))
        schema = pa.schema(
            [
                pa.field("row_id", pa.int64()),
                pa.field("i", pa.int32()),
                pa.field("j", pa.int32()),
                blob_field(col),
            ]
        )
        table = pa.table(
            {
                "row_id": pa.array(rid_list, type=pa.int64()),
                "i": pa.array(i_list, type=pa.int32()),
                "j": pa.array(j_list, type=pa.int32()),
                col: blob_array(blob_bytes),
            },
            schema=schema,
        )
        self._ds.merge_insert(["i", "j"]).when_matched_update_all().execute(table)
        self._ds = lance.dataset(self._ds.uri)
        self._coord_to_row = _load_coord_mapping(self._ds)

    def __setitem__(self, key: Any, value: Any) -> None:
        """Write through basic indices only (``r+`` mode).

        Each affected tile is read, patched in memory, re-encoded, and merged back
        on ``(i, j)`` keys.

        Parameters
        ----------
        key
            ``int`` or ``slice`` with **step 1** on both axes (after NumPy
            normalization). Fancy and boolean assignment are not supported.
        value
            Value or array broadcastable to the overlapping region.

        Raises
        ------
        NotImplementedError
            If the view is read-only, or ``key`` uses fancy/boolean indexing.
        """
        if self._encode_tile is None:
            raise NotImplementedError(
                "LanceArray is read-only; open with mode='r+' for slice assignment, "
                "or use LanceArray.to_lance() to write a full raster."
            )
        r_spec, c_spec = _normalize_lance_key(key)
        h, w = self.shape
        if (
            _coerce_1d_bool_index(r_spec) is not None
            or _coerce_1d_bool_index(c_spec) is not None
        ):
            raise NotImplementedError(
                "boolean mask assignment is not implemented; use contiguous slices"
            )
        r_cls = _classify_axis(r_spec, h, kind="row")
        c_cls = _classify_axis(c_spec, w, kind="column")
        if r_cls[0] == "adv" or c_cls[0] == "adv":
            raise NotImplementedError(
                "fancy index assignment is not implemented; use int/slice keys only"
            )
        Ri, Ci = _basic_setitem_meshes(r_cls, c_cls)
        if Ri.size == 0:
            return
        V = np.broadcast_to(np.asarray(value, dtype=self.dtype), Ri.shape)
        ch0, ch1 = self.chunks
        ij_to_stored = _stored_row_id_by_tile(self._ds)
        tile_keys = {
            (int(r) // ch0, int(c) // ch1)
            for r, c in zip(Ri.ravel(), Ci.ravel(), strict=True)
        }
        tiles_work: dict[tuple[int, int], np.ndarray] = {}
        col = self._blob_column
        for ti, tj in tile_keys:
            pos = self._coord_to_row[(ti, tj)]
            raw = self._ds.take_blobs(col, indices=[pos])[0].read()
            tiles_work[(ti, tj)] = self._decode_tile(raw).copy()
        for idx in np.ndindex(Ri.shape):
            r, c = int(Ri[idx]), int(Ci[idx])
            ti, tj = r // ch0, c // ch1
            R0, C0 = ti * ch0, tj * ch1
            tiles_work[(ti, tj)][r - R0, c - C0] = V[idx]
        self._merge_update_tiles(tiles_work, ij_to_stored_row_id=ij_to_stored)

    def to_numpy(self) -> np.ndarray:
        """Decode all tiles and return the full raster (single batched read path).

        Returns
        -------
        numpy.ndarray
            Shape ``self.shape``, dtype ``self.dtype``.
        """
        h, w = self.shape
        return self._read_subarray(0, h, 0, w)
