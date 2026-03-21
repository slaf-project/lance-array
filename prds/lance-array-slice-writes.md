# PRD: Slice assignment (`__setitem__`) for LanceArray

## Background

`LanceArray` maps a 2D raster to one Lance row per chunk tile (`i`, `j` tile indices + blob). Reads already support NumPy-style indexing including fancy indexing and boolean masks.

Lance supports efficient in-place-style updates via **merge insert** keyed on columns (e.g. `["i", "j"]`) with `when_matched_update_all()`, as described in the [Lance read/write guide — bulk update](https://lance.org/guide/read_and_write/#bulk-update).

**Bug fix (prerequisite):** `take_blobs(indices=…)` uses **positional row indices** in dataset table order, not the `row_id` column. After a merge insert, updated rows can move to the end of the table while `row_id` values are preserved, so `(i,j) → row_id` is wrong for blob reads. The coordinate map must be **`(i, j) → positional index`** derived from `to_table(columns=["i","j"])` row order.

## Goals

1. **Correct mapping** — `_load_coord_mapping` uses positional indices so reads stay correct after merge-based writes.
2. **Writable mode** — `LanceArray.open(path, mode="r+")` and `open_array(..., mode="r+")` return an array that supports assignment for **basic** indices only (see below).
3. **Slice assignment** — For allowed keys, implement `__setitem__` by read–modify–encode per touched tile, then one `merge_insert(["i","j"]).when_matched_update_all()` batch per call.
4. **Semantics** — Assignment shapes and broadcasting match NumPy for the supported index patterns.
5. **Tests** — Unit tests for single-tile and multi-tile writes, partial tiles, scalar assignment, `r` vs `r+`, and mapping correctness after a write.

## Non-goals (this iteration)

- Fancy / boolean / integer-array **assignment** (mirror read-side fancy indexing). Raise `NotImplementedError` with a clear message.
- Strided slice assignment where step ≠ 1 (optional follow-up; same machinery as step 1 but more edge cases).
- Distributed locking or multi-writer concurrency guarantees.
- Changing default `to_lance()` return value to writable; callers use `open(..., mode="r+")` to mutate on disk.

## API

| Entry | Behavior |
|-------|----------|
| `open(..., mode="r")` | Read-only; `__setitem__` raises `NotImplementedError`. |
| `open(..., mode="r+")` | Writable; encoder rebuilt from manifest (same codec params as write). |
| `__setitem__(key, value)` | See “Supported keys”. |

## Supported keys for assignment

- `int` or `slice` on each axis (after `_normalize_lance_key` / ellipsis rules).
- **Step must be 1** on both slices; otherwise raise `NotImplementedError`.
- Empty slices: no-op (no Lance write).

## Implementation notes

1. Classify axes with existing `_classify_axis`; reject `adv` on either axis for setitem.
2. Build row/col index meshes `(Ri, Ci)` with the same shapes NumPy would use for the LHS (meshgrid / full slices), then `np.broadcast_to` for `value`.
3. For each distinct `(ti, tj)` intersecting those pixels: decode tile, copy updated pixels, encode, collect merge row with columns `row_id`, `i`, `j`, `blob` — `row_id` taken from a single `to_table(columns=["row_id","i","j"])` snapshot (column value for that `(i,j)` row), not the positional index.
4. After successful `execute`, replace `self._ds` with `lance.dataset(self._ds.uri)` and refresh `self._coord_to_row` via fixed `_load_coord_mapping`.

## Success criteria

- All existing tests pass.
- New tests cover write + read-back, multi-tile updates, and `mode="r"` still read-only.
- Documentation strings for `open` / `open_array` mention `r+`.
