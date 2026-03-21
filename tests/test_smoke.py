"""Sanity check that the public package surface imports."""

from __future__ import annotations

from lance_array import LanceArray, TileCodec, normalize_chunk_slices


def test_public_exports() -> None:
    assert callable(normalize_chunk_slices)
    assert TileCodec.RAW.value == "raw"
    assert hasattr(LanceArray, "to_lance")
    assert hasattr(LanceArray, "open")
