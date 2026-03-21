"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from lance_array import LanceArray, TileCodec


@pytest.fixture
def img_and_view(tmp_path):
    """64×48 uint16 raster, 16×12 chunks, raw codec."""
    h, w = 64, 48
    ch = (16, 12)
    rng = np.random.default_rng(42)
    image = rng.integers(0, 65535, size=(h, w), dtype=np.uint16)
    path = tmp_path / "t.lance"
    view = LanceArray.to_lance(path, image, ch, codec=TileCodec.RAW)
    return image, view
