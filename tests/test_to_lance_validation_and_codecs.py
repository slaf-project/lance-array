"""``LanceArray.to_lance`` validation and codec presets."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from lance_array import LanceArray, TileCodec


def test_string_codec_aliases_raw(tmp_path) -> None:
    img = np.arange(16, dtype=np.uint16).reshape(4, 4)
    for i, codec in enumerate(("raw", "RAW", "Raw")):
        # Distinct dirs: case-only paths collide on case-insensitive filesystems.
        p = tmp_path / str(i) / "d.lance"
        view = LanceArray.to_lance(p, img, (2, 2), codec=codec)
        assert_array_equal(view.to_numpy(), img)


def test_string_codec_blosc_numcodecs_hyphen(tmp_path) -> None:
    img = np.ones((8, 8), dtype=np.uint16) * 1234
    view = LanceArray.to_lance(
        tmp_path / "b.lance",
        img,
        (4, 4),
        codec="blosc-numcodecs",
        blosc_clevel=1,
    )
    assert_array_equal(view.to_numpy(), img)


def test_unknown_codec_raises(tmp_path) -> None:
    img = np.ones((4, 4), dtype=np.uint16)
    with pytest.raises(ValueError, match="unknown tile codec"):
        LanceArray.to_lance(tmp_path / "x.lance", img, (2, 2), codec="not_a_codec")


def test_image_must_be_2d(tmp_path) -> None:
    with pytest.raises(ValueError, match="2D"):
        LanceArray.to_lance(
            tmp_path / "x.lance",
            np.ones((2, 2, 2), dtype=np.uint16),
            (1, 1),
        )


def test_shape_must_be_divisible_by_chunk(tmp_path) -> None:
    with pytest.raises(ValueError, match="divisible"):
        LanceArray.to_lance(
            tmp_path / "x.lance",
            np.ones((5, 4), dtype=np.uint16),
            (2, 2),
        )


def test_morton_tile_order_roundtrip_and_column(tmp_path) -> None:
    img = np.arange(64, dtype=np.uint16).reshape(8, 8)
    view = LanceArray.to_lance(
        tmp_path / "morton.lance",
        img,
        (2, 2),
        codec=TileCodec.RAW,
        tile_order="morton",
    )
    assert_array_equal(view.to_numpy(), img)
    names = view.dataset.schema.names
    assert "morton_code" in names


def test_hilbert_tile_order_roundtrip(tmp_path) -> None:
    img = np.arange(64, dtype=np.uint16).reshape(8, 8)
    view = LanceArray.to_lance(
        tmp_path / "hilbert.lance",
        img,
        (2, 2),
        codec=TileCodec.RAW,
        tile_order="hilbert",
    )
    assert_array_equal(view.to_numpy(), img)


def test_blosc_numcodecs_roundtrip_and_slice(tmp_path) -> None:
    rng = np.random.default_rng(7)
    img = rng.integers(0, 65535, size=(32, 24), dtype=np.uint16)
    view = LanceArray.to_lance(
        tmp_path / "nc.lance",
        img,
        (8, 6),
        codec=TileCodec.BLOSC_NUMCODECS,
        blosc_clevel=3,
        blosc_cname="zstd",
    )
    assert_array_equal(view.to_numpy(), img)
    assert_array_equal(view[5:20, 3:17], img[5:20, 3:17])


def test_blosc2_roundtrip_and_slice(tmp_path) -> None:
    pytest.importorskip("blosc2", reason="needs blosc2 (e.g. lance-array[zarr])")
    rng = np.random.default_rng(8)
    img = rng.integers(0, 65535, size=(24, 16), dtype=np.uint16)
    view = LanceArray.to_lance(
        tmp_path / "b2.lance",
        img,
        (8, 8),
        codec=TileCodec.BLOSC2,
        blosc_clevel=4,
        blosc_cname="zstd",
    )
    assert_array_equal(view.to_numpy(), img)
    assert_array_equal(view[3:15, 2:14], img[3:15, 2:14])


def test_blosc2_blosc_numcodecs_uint16_compression_ratio() -> None:
    """uint16 tiles: same typesize (2) and high enough zstd level → similar blob sizes."""
    blosc2 = pytest.importorskip(
        "blosc2", reason="needs blosc2 (e.g. lance-array[zarr])"
    )
    from lance_array.core import _build_tile_codecs

    rng = np.random.default_rng(11)
    u8 = rng.integers(0, 256, size=(128, 128), dtype=np.uint8)
    tile = (u8.astype(np.uint32) * 257).astype(np.uint16)
    chunk_shape = tile.shape
    dtype = np.dtype(np.uint16)
    itemsize = int(dtype.itemsize)
    assert itemsize == 2

    enc_nc, _ = _build_tile_codecs(
        chunk_shape,
        dtype,
        TileCodec.BLOSC_NUMCODECS,
        blosc_typesize=None,
        blosc_clevel=6,
        blosc_cname="zstd",
    )
    enc_b2, _ = _build_tile_codecs(
        chunk_shape,
        dtype,
        TileCodec.BLOSC2,
        blosc_typesize=None,
        blosc_clevel=6,
        blosc_cname="zstd",
    )
    blob_nc = enc_nc(tile)
    blob_b2 = enc_b2(tile)
    assert len(blob_nc) < tile.nbytes
    assert len(blob_b2) < tile.nbytes
    # Blosc1 vs Blosc2 framing differs slightly; stay within a small band.
    assert abs(len(blob_b2) - len(blob_nc)) / len(blob_nc) < 0.02

    wrong_ts = blosc2.compress(
        tile.tobytes(),
        typesize=8,
        clevel=6,
        filter=blosc2.Filter.SHUFFLE,
        codec=blosc2.Codec.ZSTD,
    )
    right_ts = blosc2.compress(
        tile.tobytes(),
        typesize=itemsize,
        clevel=6,
        filter=blosc2.Filter.SHUFFLE,
        codec=blosc2.Codec.ZSTD,
    )
    assert len(right_ts) <= len(wrong_ts)
