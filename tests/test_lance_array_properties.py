"""Public attributes and helpers on ``LanceArray``."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal


def _tile_from_image(
    image: np.ndarray, ti: int, tj: int, ch: tuple[int, int]
) -> np.ndarray:
    ch0, ch1 = ch
    r0, c0 = ti * ch0, tj * ch1
    return image[r0 : r0 + ch0, c0 : c0 + ch1]


def test_shape_chunks_dtype(img_and_view) -> None:
    image, view = img_and_view
    assert view.shape == image.shape
    assert view.shape == (64, 48)
    assert view.ndim == 2
    assert view.chunks == (16, 12)
    assert view.dtype == np.dtype(np.uint16)


def test_tile_grid_counts(img_and_view) -> None:
    _, view = img_and_view
    assert view.n_tile_rows == 64 // 16
    assert view.n_tile_cols == 48 // 12


def test_coord_to_row_covers_grid(img_and_view) -> None:
    _, view = img_and_view
    mapping = view.coord_to_row
    assert len(mapping) == view.n_tile_rows * view.n_tile_cols
    for ti in range(view.n_tile_rows):
        for tj in range(view.n_tile_cols):
            assert (ti, tj) in mapping
            assert isinstance(mapping[(ti, tj)], int)


def test_blob_column_and_dataset(img_and_view) -> None:
    _, view = img_and_view
    assert view.blob_column == "payload"
    assert view.dataset is not None


def test_decode_tile_single_payload(img_and_view) -> None:
    image, view = img_and_view
    rid = view.coord_to_row[(0, 0)]
    try:
        raw = view.dataset.take_blobs(view.blob_column, indices=[rid])[0].read()
    except ValueError:
        raw = view.dataset.take(indices=[rid], columns=[view.blob_column])[view.blob_column][
            0
        ].as_py()
    tile = view.decode_tile(raw)
    assert tile.shape == view.chunks
    assert_array_equal(tile, _tile_from_image(image, 0, 0, view.chunks))


def test___setitem___read_only(img_and_view) -> None:
    _, view = img_and_view
    with pytest.raises(NotImplementedError, match="read-only"):
        view[0:16, 0:12] = np.zeros((16, 12), dtype=np.uint16)
