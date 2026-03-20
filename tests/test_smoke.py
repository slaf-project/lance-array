"""Lightweight tests so CI has a green baseline; expand with Lance/Zarr fixtures later."""

from scripts.lance_vs_zarr.chunk_tiles import normalize_chunk_slices


def test_normalize_chunk_slices() -> None:
    assert normalize_chunk_slices(slice(0, 4), 10) == (0, 4)
    assert normalize_chunk_slices(slice(2, 8), 20) == (2, 8)
