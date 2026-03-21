"""Round-trip :meth:`LanceArray.open` / :func:`open_array` vs :meth:`to_lance`."""

from __future__ import annotations

import io

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from lance_array import LanceArray, TileCodec, open_array


def test_open_roundtrip_raw(tmp_path) -> None:
    img = np.arange(16, dtype=np.uint16).reshape(4, 4)
    p = tmp_path / "d.lance"
    LanceArray.to_lance(p, img, (2, 2), codec=TileCodec.RAW)
    v = LanceArray.open(p)
    assert v.ndim == 2
    assert_array_equal(v.to_numpy(), img)
    assert_array_equal(open_array(p)[:, :], img)


def test_open_roundtrip_blosc_numcodecs(tmp_path) -> None:
    img = np.ones((8, 8), dtype=np.uint16) * 999
    p = tmp_path / "b.lance"
    LanceArray.to_lance(p, img, (4, 4), codec=TileCodec.BLOSC_NUMCODECS, blosc_clevel=1)
    v = open_array(p, mode="r")
    assert_array_equal(v[0:4, 0:4], np.full((4, 4), 999, dtype=np.uint16))


def test_open_invalid_mode(tmp_path) -> None:
    p = tmp_path / "d.lance"
    LanceArray.to_lance(p, np.zeros((2, 2), dtype=np.uint16), (1, 1))
    with pytest.raises(ValueError, match="invalid mode"):
        LanceArray.open(p, mode="w")


def test_open_s3_uri_reads_manifest_via_smart_open(tmp_path, monkeypatch) -> None:
    pytest.importorskip("smart_open")
    import lance as lance_mod

    from lance_array.core import LanceArray, TileCodec

    img = np.arange(16, dtype=np.uint16).reshape(4, 4)
    p = tmp_path / "d.lance"
    LanceArray.to_lance(p, img, (2, 2), codec=TileCodec.RAW)
    manifest_text = (p / "lance_array.json").read_text(encoding="utf-8")
    real_ds = lance_mod.dataset(str(p))
    urls: list[str] = []

    def fake_smart_open(url: str, *a, **kw):
        urls.append(url)
        if url.endswith("lance_array.json"):
            return io.StringIO(manifest_text)
        raise AssertionError(f"unexpected url {url!r}")

    monkeypatch.setattr("smart_open.open", fake_smart_open)
    monkeypatch.setattr("lance_array.core.lance.dataset", lambda uri: real_ds)

    v = LanceArray.open("s3://bucket/prefix/d.lance")
    assert urls == ["s3://bucket/prefix/d.lance/lance_array.json"]
    assert_array_equal(v.to_numpy(), img)
