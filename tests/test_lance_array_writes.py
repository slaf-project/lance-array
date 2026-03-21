"""Slice assignment with ``mode='r+'`` and merge-insert tile updates."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from lance_array import LanceArray, TileCodec, open_array


def test_setitem_requires_r_plus(tmp_path) -> None:
    img = np.arange(16, dtype=np.uint16).reshape(4, 4)
    p = tmp_path / "d.lance"
    LanceArray.to_lance(p, img, (2, 2), codec=TileCodec.RAW)
    ro = LanceArray.open(p, mode="r")
    with pytest.raises(NotImplementedError, match="read-only"):
        ro[0:2, 0:2] = np.zeros((2, 2), dtype=np.uint16)  # type: ignore[misc]


def test_setitem_single_tile_full(tmp_path) -> None:
    img = np.arange(16, dtype=np.uint16).reshape(4, 4)
    p = tmp_path / "d.lance"
    LanceArray.to_lance(p, img, (2, 2), codec=TileCodec.RAW)
    rw = LanceArray.open(p, mode="r+")
    patch = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    rw[0:2, 0:2] = patch
    ro = LanceArray.open(p, mode="r")
    assert_array_equal(ro[0:2, 0:2], patch)
    assert_array_equal(ro[2:4, 2:4], img[2:4, 2:4])


def test_setitem_scalar_and_row_slice(tmp_path) -> None:
    img = np.arange(16, dtype=np.uint16).reshape(4, 4)
    p = tmp_path / "d.lance"
    LanceArray.to_lance(p, img, (2, 2), codec=TileCodec.RAW)
    rw = open_array(p, mode="r+")
    expected = img.copy()
    expected[1, 0:4] = 777
    rw[1, 0:4] = 777
    ro = open_array(p)
    assert_array_equal(ro.to_numpy(), expected)


def test_setitem_multi_tile_rectangle(tmp_path) -> None:
    img = np.zeros((8, 8), dtype=np.uint16)
    p = tmp_path / "d.lance"
    LanceArray.to_lance(p, img, (4, 4), codec=TileCodec.RAW)
    rw = LanceArray.open(p, mode="r+")
    patch = np.ones((6, 6), dtype=np.uint16) * 42
    rw[1:7, 1:7] = patch
    ro = LanceArray.open(p)
    assert_array_equal(ro[1:7, 1:7], patch)


def test_setitem_partial_tile_two_chunks(tmp_path) -> None:
    h, w = 64, 48
    ch = (16, 12)
    rng = np.random.default_rng(0)
    img = rng.integers(0, 1000, size=(h, w), dtype=np.uint16)
    p = tmp_path / "big.lance"
    LanceArray.to_lance(p, img, ch, codec=TileCodec.RAW)
    rw = LanceArray.open(p, mode="r+")
    patch = rng.integers(0, 1000, size=(10, 9), dtype=np.uint16)
    rw[5:15, 7:16] = patch
    ro = LanceArray.open(p)
    assert_array_equal(ro[5:15, 7:16], patch)
    assert_array_equal(ro[0:5, :], img[0:5, :])
    assert_array_equal(ro[15:, :], img[15:, :])


def test_setitem_empty_slice_noop(tmp_path) -> None:
    img = np.arange(16, dtype=np.uint16).reshape(4, 4)
    p = tmp_path / "d.lance"
    LanceArray.to_lance(p, img, (2, 2), codec=TileCodec.RAW)
    rw = LanceArray.open(p, mode="r+")
    rw[2:2, 0:4] = np.zeros((0, 4), dtype=np.uint16)  # type: ignore[misc]
    assert_array_equal(rw.to_numpy(), img)


def test_setitem_fancy_raises(tmp_path) -> None:
    img = np.arange(16, dtype=np.uint16).reshape(4, 4)
    p = tmp_path / "d.lance"
    LanceArray.to_lance(p, img, (2, 2), codec=TileCodec.RAW)
    rw = LanceArray.open(p, mode="r+")
    with pytest.raises(NotImplementedError, match="fancy"):
        rw[[0, 1], [1, 2]] = np.array([1, 2], dtype=np.uint16)  # type: ignore[index]


def test_setitem_boolean_raises(tmp_path) -> None:
    img = np.arange(16, dtype=np.uint16).reshape(4, 4)
    p = tmp_path / "d.lance"
    LanceArray.to_lance(p, img, (2, 2), codec=TileCodec.RAW)
    rw = LanceArray.open(p, mode="r+")
    m = np.zeros(4, dtype=bool)
    m[:2] = True
    with pytest.raises(NotImplementedError, match="boolean"):
        rw[m, 0:4] = 0  # type: ignore[index]


def test_setitem_strided_slice_raises(tmp_path) -> None:
    img = np.arange(16, dtype=np.uint16).reshape(4, 4)
    p = tmp_path / "d.lance"
    LanceArray.to_lance(p, img, (2, 2), codec=TileCodec.RAW)
    rw = LanceArray.open(p, mode="r+")
    with pytest.raises(NotImplementedError, match="step"):
        rw[0:4:2, 0:4] = np.ones((2, 4), dtype=np.uint16)  # type: ignore[misc]


def test_setitem_blosc_roundtrip(tmp_path) -> None:
    img = np.arange(64, dtype=np.uint16).reshape(8, 8)
    p = tmp_path / "b.lance"
    LanceArray.to_lance(p, img, (4, 4), codec=TileCodec.BLOSC_NUMCODECS, blosc_clevel=1)
    rw = LanceArray.open(p, mode="r+")
    rw[0:4, 0:4] = np.ones((4, 4), dtype=np.uint16) * 500
    ro = LanceArray.open(p)
    assert_array_equal(ro[0:4, 0:4], 500)
    assert_array_equal(ro[4:8, 4:8], img[4:8, 4:8])


def test_coord_mapping_positional_after_write(tmp_path) -> None:
    """Regression: take_blobs uses positional indices; mapping must stay valid."""
    img = np.arange(16, dtype=np.uint16).reshape(4, 4)
    p = tmp_path / "d.lance"
    LanceArray.to_lance(p, img, (2, 2), codec=TileCodec.RAW)
    rw = LanceArray.open(p, mode="r+")
    rw[0:2, 0:2] = np.full((2, 2), 999, dtype=np.uint16)
    # Same view object must still read correctly after internal reload
    assert_array_equal(rw[0:2, 0:2], np.full((2, 2), 999, dtype=np.uint16))
    assert_array_equal(rw[2:4, 2:4], img[2:4, 2:4])
