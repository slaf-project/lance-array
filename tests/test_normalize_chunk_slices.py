"""Unit tests for ``normalize_chunk_slices``."""

from __future__ import annotations

import pytest

from lance_array import normalize_chunk_slices


def test_basic_ranges() -> None:
    assert normalize_chunk_slices(slice(0, 4), 10) == (0, 4)
    assert normalize_chunk_slices(slice(2, 8), 20) == (2, 8)


def test_omitted_bound_defaults() -> None:
    assert normalize_chunk_slices(slice(None, 5), 10) == (0, 5)
    assert normalize_chunk_slices(slice(3, None), 10) == (3, 10)
    assert normalize_chunk_slices(slice(None, None), 7) == (0, 7)


def test_negative_indices_resolved_against_dim() -> None:
    assert normalize_chunk_slices(slice(-3, None), 10) == (7, 10)
    assert normalize_chunk_slices(slice(-10, -2), 20) == (10, 18)


def test_rejects_non_unit_step() -> None:
    with pytest.raises(ValueError, match="step"):
        normalize_chunk_slices(slice(0, 8, 2), 20)
    with pytest.raises(ValueError, match="step"):
        normalize_chunk_slices(slice(None, None, -1), 10)


def test_rejects_empty_slice() -> None:
    with pytest.raises(ValueError, match="empty"):
        normalize_chunk_slices(slice(3, 3), 10)
    with pytest.raises(ValueError, match="empty"):
        normalize_chunk_slices(slice(8, 2), 10)
