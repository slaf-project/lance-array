#!/usr/bin/env python3
"""Build Zarr vs Lance bar charts from Rich-style benchmark summary text.

**Layout:** One **row per condition** (facet). **X-axis** = p50, p95, p99 (each a Zarr+Lance
pair). **Legend** is horizontal in the figure margin (below suptitle, above panels).
**Figure captions** live in `README.md` and `docs/index.md` under each image (not in the SVG).

  uv sync --extra dev
  uv run python scripts/render_benchmark_charts.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

_MS = re.compile(r"([\d.]+)\s*ms")
_SLICE_N = re.compile(r"^(\d+)\s*[×x]\s*(\d+)$", re.IGNORECASE)

_LANCE_RAW_KEYS = (
    "Lance uncompressed (Morton order)",
    "Lance uncompressed (Blosc2+Morton)",
)
_LANCE_COMP_KEYS = (
    "Lance compressed (Blosc2 + Morton)",
    "Lance compressed (Blosc2+Morton)",
)


def _parse_ms(cell: str) -> float:
    m = _MS.search(cell)
    if not m:
        return float("nan")
    return float(m.group(1))


def _row_cells(line: str, *, min_cells: int) -> list[str] | None:
    line = line.strip()
    if not line.startswith("│"):
        return None
    cells = [p.strip() for p in line.split("│")]
    cells = [c for c in cells if c]
    return cells if len(cells) >= min_cells else None


def parse_best_case_table(text: str) -> dict[tuple[str, str], dict[str, float]]:
    out: dict[tuple[str, str], dict[str, float]] = {}
    in_first = False
    for line in text.splitlines():
        if (
            "Best-case Zarr vs Lance" in line
            or "best-case Zarr vs Lance" in line.lower()
        ):
            in_first = True
            continue
        if in_first and ("Morton vs Row Ordering" in line or "Morton vs row" in line):
            break
        cells = _row_cells(line, min_cells=6)
        if not cells:
            continue
        if cells[0] == "Backend":
            continue
        backend, request, mean_s, p50_s, p95_s, p99_s = cells[:6]
        out[(backend, request)] = {
            "mean": _parse_ms(mean_s),
            "p50": _parse_ms(p50_s),
            "p95": _parse_ms(p95_s),
            "p99": _parse_ms(p99_s),
        }
    return out


def parse_slice_scaling_table(
    text: str,
) -> list[tuple[int, dict[str, float], dict[str, float]]]:
    out: list[tuple[int, dict[str, float], dict[str, float]]] = []
    in_table = False
    for line in text.splitlines():
        if "Zarr compressed vs Lance compressed" in line and "slice size" in line:
            in_table = True
            continue
        if not in_table:
            continue
        cells = _row_cells(line, min_cells=9)
        if not cells:
            if out and line.strip().startswith("└"):
                break
            continue
        head = cells[0].replace("x", "×")
        m = _SLICE_N.match(head)
        if not m or m.group(1) != m.group(2):
            if "N×N" in head or "NxN" in cells[0]:
                continue
            continue
        n = int(m.group(1))
        _z_mu, zp50, zp95, zp99 = cells[1], cells[2], cells[3], cells[4]
        _l_mu, lp50, lp95, lp99 = cells[5], cells[6], cells[7], cells[8]
        zd = {"p50": _parse_ms(zp50), "p95": _parse_ms(zp95), "p99": _parse_ms(zp99)}
        ld = {"p50": _parse_ms(lp50), "p95": _parse_ms(lp95), "p99": _parse_ms(lp99)}
        out.append((n, zd, ld))
    return out


def _pick_lance(
    table: dict[tuple[str, str], dict[str, float]],
    request: str,
    keys: tuple[str, ...],
) -> dict[str, float]:
    for k in keys:
        row = table.get((k, request))
        if row:
            return {x: row[x] for x in ("p50", "p95", "p99")}
    return {"p50": float("nan"), "p95": float("nan"), "p99": float("nan")}


def build_pairwise_series(
    text: str,
) -> list[tuple[str, dict[str, float], dict[str, float]]]:
    """List of (facet_title, zarr{p50,p95,p99}, lance{...})."""
    t1 = parse_best_case_table(text)
    slices = parse_slice_scaling_table(text)

    rows: list[tuple[str, dict[str, float], dict[str, float]]] = []

    def zarr_row(backend: str, req: str) -> dict[str, float]:
        r = t1.get((backend, req), {})
        return {k: r.get(k, float("nan")) for k in ("p50", "p95", "p99")}

    rows.append(
        (
            "Single tile (uncompressed)",
            zarr_row("Zarr uncompressed", "single"),
            _pick_lance(t1, "single", _LANCE_RAW_KEYS),
        )
    )
    rows.append(
        (
            "Single tile (compressed)",
            zarr_row("Zarr compressed", "single"),
            _pick_lance(t1, "single", _LANCE_COMP_KEYS),
        )
    )
    for n, zd, ld in sorted(slices, key=lambda x: x[0]):
        rows.append((f"Slice {n}×{n}", zd, ld))

    return rows


def render_figure(
    facets: list[tuple[str, dict[str, float], dict[str, float]]],
    *,
    suptitle: str,
    outfile: Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    percentiles = ("p50", "p95", "p99")
    x_centers = np.arange(len(percentiles), dtype=float)
    bar_w = 0.36
    z_color = "#2f5b9c"
    l_color = "#c44e52"

    n = len(facets)
    # Narrow width (only 3 percentile groups on x); taller rows per facet.
    fig_w = 4.85
    # Extra height: room for suptitle + figure-level legend above subplot block.
    fig_h = max(2.65, 1.88 * n + 1.35)
    fig, axes = plt.subplots(
        n,
        1,
        figsize=(fig_w, fig_h),
        sharex=True,
        sharey=False,
    )

    if n == 1:
        axes = [axes]

    for ax, (facet_title, zd, ld) in zip(axes, facets, strict=True):
        z_vals = [zd[p] for p in percentiles]
        l_vals = [ld[p] for p in percentiles]
        ax.bar(
            x_centers - bar_w / 2,
            z_vals,
            bar_w,
            color=z_color,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.bar(
            x_centers + bar_w / 2,
            l_vals,
            bar_w,
            color=l_color,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_ylabel("ms")
        ax.set_title(facet_title, loc="left", fontsize=10, fontweight="normal")
        ax.grid(axis="y", alpha=0.35)
        ax.set_axisbelow(True)
        ax.set_xticks(x_centers)
        ax.set_xticklabels(list(percentiles))

    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)

    legend_el = [
        Patch(facecolor=z_color, edgecolor="white", linewidth=0.5, label="Zarr"),
        Patch(facecolor=l_color, edgecolor="white", linewidth=0.5, label="Lance"),
    ]

    # Suptitle + legend in figure coordinates so the legend does not sit on the
    # first panel's facet title (axes-coords bbox_to_anchor above 1.0 collides).
    fig.suptitle(suptitle, fontsize=12, fontweight="600", y=0.97)
    fig.legend(
        handles=legend_el,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.915),
        bbox_transform=fig.transFigure,
        ncol=2,
        frameon=False,
        fontsize=9,
        columnspacing=1.8,
        handletextpad=0.5,
    )
    fig.subplots_adjust(
        left=0.18,
        right=0.98,
        top=0.78,
        bottom=0.1,
        hspace=0.52,
    )

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, format="svg", bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--local", type=Path, default=Path("scripts/local_summary.txt"))
    ap.add_argument("--s3", type=Path, default=Path("scripts/s3_summary.txt"))
    ap.add_argument("--out-dir", type=Path, default=Path("docs/images/benchmarks"))
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]

    def load(p: Path) -> str:
        path = p if p.is_absolute() else root / p
        return path.read_text(encoding="utf-8")

    out = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir

    for key, path, name in (
        ("local", args.local, "benchmark_local_p50_p95_p99.svg"),
        ("s3", args.s3, "benchmark_s3_p50_p95_p99.svg"),
    ):
        text = load(path)
        facets = build_pairwise_series(text)
        render_figure(
            facets,
            suptitle=(
                "Zarr vs Lance · local SSD"
                if key == "local"
                else "Zarr vs Lance · object store → laptop"
            ),
            outfile=out / name,
        )

    print(
        f"Wrote:\n  {out / 'benchmark_local_p50_p95_p99.svg'}\n  {out / 'benchmark_s3_p50_p95_p99.svg'}"
    )


if __name__ == "__main__":
    main()
