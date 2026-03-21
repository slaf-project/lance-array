"""LanceArray 2D indexing vs NumPy semantics (raw codec, temp dataset)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal


def test_to_numpy_roundtrip(img_and_view):
    image, view = img_and_view
    assert_array_equal(view.to_numpy(), image)


def test_arbitrary_slice_matches_numpy(img_and_view):
    image, view = img_and_view
    assert_array_equal(view[5:37, 7:41], image[5:37, 7:41])
    assert_array_equal(view[0:16, 0:12], image[0:16, 0:12])
    assert_array_equal(view[15:50, 11:40], image[15:50, 11:40])


def test_int_and_mixed_indexing(img_and_view):
    image, view = img_and_view
    assert_array_equal(view[3, 4], np.array(image[3, 4], dtype=np.uint16))
    assert_array_equal(view[3, 4:30], image[3, 4:30])
    assert_array_equal(view[10:44, 6], image[10:44, 6])


def test_negative_index(img_and_view):
    image, view = img_and_view
    assert_array_equal(view[-1, -3], np.array(image[-1, -3], dtype=np.uint16))
    assert_array_equal(view[-5:-1, -8:-2], image[-5:-1, -8:-2])


def test_empty_slices(img_and_view):
    image, view = img_and_view
    assert_array_equal(view[10:10, 5:20], image[10:10, 5:20])
    assert_array_equal(view[0:10, 3:3], image[0:10, 3:3])
    assert_array_equal(view[2:2, 9:9], image[2:2, 9:9])


def test_stepped_slice_matches_numpy(img_and_view):
    image, view = img_and_view
    assert_array_equal(view[0:10:2, 0:12:3], image[0:10:2, 0:12:3])
    assert_array_equal(view[1:17:4, 3:40:5], image[1:17:4, 3:40:5])


def test_row_only_and_ellipsis_match_numpy(img_and_view):
    image, view = img_and_view
    assert_array_equal(view[7], image[7])
    assert_array_equal(view[7,], image[7,])
    assert_array_equal(view[8:20], image[8:20])
    assert_array_equal(view[...], image[...])
    assert_array_equal(view[..., 5:30], image[..., 5:30])
    assert_array_equal(view[3:15, ...], image[3:15, ...])


def test_ix_and_fancy_matches_numpy(img_and_view):
    image, view = img_and_view
    ri, ci = np.ix_([2, 5, 11], [1, 4, 7])
    assert_array_equal(view[ri, ci], image[ri, ci])
    assert_array_equal(view[[10, 12, 14], 6:35], image[[10, 12, 14], 6:35])
    assert_array_equal(view[4:40, [3, 6, 9]], image[4:40, [3, 6, 9]])
    assert_array_equal(view[[1, 2, 3], [10, 20, 30]], image[[1, 2, 3], [10, 20, 30]])


def test_boolean_mask_matches_numpy(img_and_view):
    image, view = img_and_view
    mr = np.zeros(image.shape[0], dtype=bool)
    mr[3:9] = True
    mc = np.zeros(image.shape[1], dtype=bool)
    mc[10:22:2] = True
    assert_array_equal(view[mr, :], image[mr, :])
    assert_array_equal(view[:, mc], image[:, mc])
    assert_array_equal(view[mr, mc], image[mr, mc])


def test_boolean_masks_broadcast_like_numpy(img_and_view):
    image, view = img_and_view
    mr = np.zeros(image.shape[0], dtype=bool)
    mr[20] = True
    mc = np.zeros(image.shape[1], dtype=bool)
    mc[5:15] = True
    assert_array_equal(view[mr, mc], image[mr, mc])


def test_boolean_masks_incompatible_raises(img_and_view):
    _, view = img_and_view
    mr = np.zeros(view.shape[0], dtype=bool)
    mr[:10] = True
    mc = np.zeros(view.shape[1], dtype=bool)
    mc[:7] = True
    with pytest.raises(IndexError, match="broadcast"):
        _ = view[mr, mc]


def test_reject_python_bool_scalar(img_and_view):
    _, view = img_and_view
    with pytest.raises(TypeError, match="numpy"):
        view[True, 0:3]  # type: ignore[index]
    with pytest.raises(TypeError, match="numpy"):
        view[0:5, False]  # type: ignore[index]


def test_too_many_indices(img_and_view):
    _, view = img_and_view
    with pytest.raises(IndexError, match="too many indices"):
        view[0, 1, 2]  # type: ignore[index]


def test_index_error(img_and_view):
    _, view = img_and_view
    with pytest.raises(IndexError):
        _ = view[100, 0]
    with pytest.raises(IndexError):
        _ = view[0, -100]
