# PRD: N-dimensional arrays (extend beyond 2D)

## Background

`LanceArray` is a **chunk-aligned view** over a Lance dataset: **one Lance row per tile**, with tile coordinates **`(i, j)`**, a blob payload per row, and a sidecar manifest (`lance_array.json`) for `shape`, `chunk_shape`, dtype, and codec metadata.

The implementation is **specialized to 2D** end-to-end:

- **Schema:** columns `row_id`, `i`, `j`, and the blob column; merge-upsert keys are `["i", "j"]`.
- **Mapping:** `coord_to_row: dict[tuple[int, int], int]` from tile grid coordinates to **positional** row index for `take_blobs` (see existing PRD on positional vs `row_id`).
- **Reads:** `_read_subarray(r0, r1, c0, c1)` implements an axis-aligned **rectangle** in row/col space, iterating overlapping tiles in a **2D nested loop** and copying from **2D** decoded tiles into the output buffer.
- **Indexing:** `_normalize_lance_key` and helpers assume **two axes** (ellipsis expansion to length 2). Advanced indexing and boolean masks are built for **row/column** semantics (`_combine_for_advanced`, `_gather_at_pairs`, two 1D boolean masks).
- **Writes:** `__setitem__` builds **2D** index meshes `(Ri, Ci)` and groups updates by tile key `(ti, tj)`.

**Tile codecs** (`TileCodec`, `_build_tile_codecs`) already treat a tile as **flat bytes** reshaped to `chunk_shape`; an n-tuple `chunk_shape` is conceptually compatible without changing the byte layout.

This PRD describes what it would take to support **`ndim ≥ 2`** (and typically **`ndim` arbitrary but fixed per dataset**) without prescribing a single release slice; implementation may be phased.

## Goals

1. **On-disk representation** — Represent an n-dimensional logical array: manifest stores **`shape` and `chunk_shape` as length-n lists**; each tile row stores an **n-dimensional tile index** (see “Schema options” below).
2. **Round-trip I/O** — `to_lance` (or equivalent) can write an **`ndim`-D NumPy array** whose shape is divisible by `chunk_shape` per axis; `open` / `open_array` reconstruct a view with correct `shape`, `chunks`, dtype, and codec behavior.
3. **Core read path** — Read any **axis-aligned hyperrectangle** (half-open intervals per axis) by determining overlapping tiles along each axis, iterating the **Cartesian product** of tile index ranges, batching `take_blobs`, and stitching **n-dimensional** slices from each decoded tile into the output array.
4. **Public API** — `ndim` reflects `len(shape)`; per-axis tile counts or a single `tile_grid_shape` tuple replace or generalize `n_tile_rows` / `n_tile_cols` as needed.
5. **Backward compatibility** — Existing **2D** datasets (current manifest and `i`, `j` columns) continue to open and behave as today, either unchanged or via explicit format version branching.

## Non-goals (initial / optional)

- **Full NumPy advanced indexing parity in nD** in the first iteration (broadcasting of integer arrays, mixed basic/advanced rules, etc.) — high complexity relative to the 2D special case.
- **All boolean indexing patterns** that NumPy allows for nD (e.g. single mask matching full shape vs separate masks per axis) — may be deferred or subsetted with clear errors.
- **Distributed locking / multi-writer** semantics (same as 2D today).
- **Changing** Lance’s fundamental row-per-tile model (e.g. one row per file) — out of scope unless separately motivated.

## Schema options (tile coordinates)

Choose one approach and document it in the manifest / format version:

| Approach | Pros | Cons |
|----------|------|------|
| **Fixed columns** `i0`, `i1`, …, `i{n-1}` | Simple types, straightforward merge keys | Schema width grows with max supported `ndim`; may need a cap or nullable columns for migration |
| **Single list column** for tile index | Natural variable `ndim` | Arrow/Lance ergonomics for merge keys and queries need validation |
| **Struct column** | Structured index in one field | Same merge-key and tooling questions as list |

Merge insert must key on the **full tile index** (generalization of current `["i", "j"]`).

## Indexing semantics (phased)

### Phase A (recommended first)

- **`__getitem__`:** `int`, `slice` (consider **step 1 only** first), `...`, and NumPy’s **ellipsis fill rule** generalized to **`ndim`**: `n_fill = ndim - len(before) - len(after)` instead of hard-coded `2`.
- **Omitted trailing axes:** Match NumPy: single index applies to axis 0, remaining axes full slice (generalization of current `key` → `(key, slice(None))` behavior).
- **`__setitem__`:** Same restricted basic indexing as Phase A reads, with **step 1** slices if that matches the 2D write PRD trajectory.
- **Explicit errors:** `NotImplementedError` or `IndexError` with clear messages for fancy/boolean/strided patterns until implemented.

### Phase B (optional, large effort)

- Port **advanced indexing** to nD with NumPy-compatible broadcasting rules (generalization of `_combine_for_advanced`, `_gather_at_pairs`).
- Define supported **boolean** patterns (e.g. only a tuple of `ndim` 1D masks vs full-shaped mask) and implement or reject explicitly.

### Implementation note on strided slices

The 2D codebase has a dedicated strided read path (`_slice_slice_strided`). nD requires an analogous **n-axis strided** reader (or a bounding-box read plus in-memory strided extraction, trading IO for simplicity).

## Core algorithm sketch: `_read_subarray` in nD

Input: per-axis half-open ranges `[start[d], stop[d])` for `d = 0..ndim-1`.

1. Allocate output array with shape `(stop[d] - start[d])` for each `d`.
2. For each axis `d`, compute tile index range from chunk size `chunks[d]` (same formulas as current `i_start`, `i_end`, `j_start`, `j_end`).
3. For each **tile index vector** in the Cartesian product of those ranges:
   - Resolve Lance row via `coord_to_row[tile_index_tuple]`.
   - Decode tile; compute **intersection** of global request with tile’s global bounds as **n pairs** of source slices into the tile and dest slices into `out` (direct generalization of current `tr0/tr1/tc0/tc1` / `sr/sc/dr/dc` logic).
4. Batch `take_blobs` over unique row positions as today.

## Manifest and format versioning

- Extend manifest payload with **`ndim`** and/or bump **`MANIFEST_VERSION`** so readers can distinguish legacy 2D (`shape` length 2, columns `i`, `j`) from nD layouts.
- Validation on open: product of tile counts matches row count; `shape[d] % chunks[d] == 0` for all `d`.

## Tile codecs

No fundamental change: encode/decode still **reshape** decoded bytes to `chunk_shape` (now length `n`). Verify blosc/raw size checks use **product of `chunk_shape` × itemsize**.

## Writes (`__setitem__`)

- Generalize **basic** assignment meshes from `(Ri, Ci)` to **n coordinate arrays** (or equivalent) consistent with NumPy LHS shape rules for the supported patterns.
- Tile keys become **`tuple[int, ...]`** of length `ndim`; read–modify–encode per touched tile unchanged in spirit.
- `_merge_update_tiles` (or successor) emits **all tile coordinate columns** and merge-keys them consistently with the schema choice.

## Success criteria

- Documented **format** and **compatibility story** for existing 2D datasets.
- Phase A: at least one **3D** (or arbitrary small **n > 2**) integration test: write → open → slice read → slice write (if in scope) → read-back matches NumPy ground truth.
- Phase A: errors for unsupported indexing are **explicit** (not silent wrong results).
- Optional Phase B: tests aligned with chosen NumPy subset for advanced/boolean indexing.

## References

- Existing slice-write PRD: `prds/lance-array-slice-writes.md` (positional `coord_to_row`, merge insert, basic `__setitem__` constraints).
- Lance bulk update pattern: [Lance read/write guide — bulk update](https://lance.org/guide/read_and_write/#bulk-update).
