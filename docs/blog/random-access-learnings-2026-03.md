# Random Access Learnings (March 2026)

This note summarizes the random-access benchmarking work done for `lance-array` and the key decisions that followed.

## Scope and setup

- Focused benchmark target: 2D tile slicing latency on local SSD and S3.
- Dedicated slice-scaling harness: `scripts/run_slice_scaling_benchmark.py`.
- Compared:
  - Zarr Blosc + `zarrs` Rust pipeline
  - Lance Blosc2 (blob column, row order)
  - Lance Blosc2 (blob column, Morton order)
  - Lance bytes + Blosc2 (row order)
  - Lance bytes + Blosc2 (Morton order)
- Slice scales: `1x1`, `3x3`, `5x5`, `7x7`, `9x9` tiles.
- Chunk shape for scaling experiments: `64x64`.

## Step-by-step findings

### 1) Manifest pinning and tile index are not enough for slice parity

- Manifest/index pinning reduced control-plane overhead but did not close slice latency gaps.
- Single-tile benchmarks were noisy and close enough that differences were not strongly actionable.

### 2) Morton write ordering alone did not guarantee slice wins

- Morton ordering helped in some runs and scales, but was inconsistent.
- With blob-backed payloads, Zarr still won most slice scales.

### 3) Decode concurrency had negligible net benefit

- Added threaded `read()+decode` in the Lance read path.
- Result: little or no consistent win; fetch still dominated in observed workloads.

### 4) Blob vs bytes payload path was the biggest unlock

- Blob-backed path (`take_blobs` + `BlobFile.read`) carries per-item Python/file-object overhead.
- Bytes-backed path (`take(..., columns=["payload"])`) returned compressed payloads directly to memory.
- In local slice-scaling runs, bytes-backed Lance substantially outperformed blob-backed Lance and often beat Zarr.

## Latest reference run (local SSD)

Command:

```bash
uv run python scripts/run_slice_scaling_benchmark.py --recreate --repeats 2 --seed 42
```

Configuration:

- Chunk shape: `64x64`
- Reads per repeat: `500`
- Slices: `1x1`, `3x3`, `5x5`, `7x7`, `9x9`
- Percentiles reported from repeated runs

Reference means (ms):

| Slice scale | Zarr (Blosc + zarrs) | Lance (Blosc2, blob row) | Lance (Blosc2, blob Morton) | Lance (bytes+Blosc2, row) | Lance (bytes+Blosc2, Morton) |
|---|---:|---:|---:|---:|---:|
| `1x1` | 0.466 | 0.576 | 0.507 | 0.376 | 0.360 |
| `3x3` | 0.813 | 1.748 | 2.142 | 0.510 | 0.504 |
| `5x5` | 1.566 | 3.352 | 3.531 | 0.713 | 0.706 |
| `7x7` | 1.901 | 5.492 | 5.462 | 1.017 | 1.005 |
| `9x9` | 2.929 | 8.002 | 7.018 | 1.758 | 1.487 |

Primary reminder from this run:

- On local SSD, **Lance with raw bytes + Morton ordering is about 2x faster than Zarr** at larger slice scales (`9x9`: `1.487ms` vs `2.929ms`).

## Storage and compression observations

- Chunk size strongly changed storage behavior and results.
- At `64x64`, Lance variants compressed to about `1.9 MiB` while Zarr Blosc was around `4 MiB` on the same source image.
- At larger chunk sizes (for example `256x256` in earlier full-suite runs), sizes were much closer.
- Conclusion: compression ratio is chunk-size dependent for both systems; benchmark interpretation must keep chunk size fixed.

## Schema observations and follow-ups

Current tile schema fields include:

- `row_id` (`int64`)
- tile coordinates (`i`, `j`) as `int32`

Open design questions:

- Can `row_id` be removed (or made optional) for read-only array workloads?
- Should tile coordinates be `uint32` instead of `int32` (no negative tile coordinates expected)?
- Should coordinate columns be renamed from `i`,`j` to clearer names (for example `tile_row`,`tile_col`)?

These are compatibility and ergonomics questions that should be benchmarked and validated before any default changes.

## What this implies for the roadmap

- For tile-array workloads that always decode full payloads, bytes payload columns are a high-value direction.
- Blob V2 remains valuable for mixed/larger object use cases and lazy/range read semantics.
- The next optimization step should be chosen per workload:
  - Tile arrays: prioritize bytes payload fast path and fetch behavior.
  - Mixed blob workloads: continue with Blob V2 semantics.

## References

- Lance Blob V2 overview: [Lance Blob V2: Making Multimodal Data a First-Class Citizen in the Lakehouse](https://lancedb.com/blog/lance-blob-v2/)
- Lance scalar index spec (R-Tree): [RTree - Lance](https://lance.org/format/table/index/scalar/rtree/)
