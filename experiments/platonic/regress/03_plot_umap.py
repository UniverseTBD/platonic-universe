#!/usr/bin/env python3
"""Step 3 — UMAP figures.

Reads every per-tuple ``umap.parquet`` produced by step 1 and renders
two output flavours, both populated by default:

  - ``umap/<modality>/<alias>_<size>__<property>.png`` — one PNG per
    (tuple, property), so a downstream analysis can grab a specific
    panel without slicing a PDF.
  - ``umap_<modality>.pdf`` (with ``--pdf``) — multi-page overview;
    one page per property, grid = family × size for at-a-glance
    structure-vs-scale comparison.

Within both outputs, every panel is plotted using the same random
subsample of galaxies so cross-panel comparison is visually fair.

Usage
-----
    python 03_plot_umap.py \\
        --pull-from "$PU_REGRESS_RESULTS_REPO" \\
        --out-dir   ./derived

    # or with a local cache already populated by step 2:
    python 03_plot_umap.py --done-dir ./derived/_regress_cache/done \\
        --out-dir ./derived
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

# Order families/sizes the same way as the rest of the pipeline so the
# plots line up with every other table in the paper.
FAMILY_SIZES: dict[str, list[str]] = {
    "vit":       ["base", "large", "huge"],
    "clip":      ["base", "large"],
    "dinov3":    ["vits16", "vits16plus", "vitb16", "vitl16",
                  "vith16plus", "vit7b16"],
    "convnext":  ["nano", "tiny", "base", "large"],
    "ijepa":     ["huge", "giant"],
    "vjepa":     ["large", "huge", "giant"],
    "astropt":   ["015M", "095M", "850M"],
    "vit-mae":   ["base", "large", "huge"],
    "paligemma": ["3b", "10b", "28b"],
    "llava_15":  ["7b", "13b"],
}
MODALITIES = ("hsc", "jwst")
PROPERTIES = ("redshift", "mag_g", "mag_r", "mass", "sSFR", "g-r")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--pull-from", default=os.environ.get("PU_REGRESS_RESULTS_REPO"),
                   help="HF dataset id to fetch umap.parquet from (skipped if "
                        "--done-dir is supplied).")
    p.add_argument("--done-dir", type=Path, default=None,
                   help="Pre-populated local mirror of done/<tag>/ (overrides --pull-from).")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Where to write the PDFs.")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="Where to keep pulled parquets (default: $out_dir/_regress_cache).")
    p.add_argument("--n-points", type=int, default=15_000,
                   help="Subsample to this many galaxies per panel (default 15k).")
    p.add_argument("--pdf", action="store_true",
                   help="Also render the multi-page PDF overview per modality.")
    p.add_argument("--no-png", action="store_true",
                   help="Skip the per-tuple PNGs (only meaningful with --pdf).")
    p.add_argument("--png-dpi", type=int, default=150,
                   help="DPI for individual PNGs.")
    return p.parse_args()


def fetch_done(repo: str, dest: Path, token: str | None) -> Path:
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=token)
    files = [f for f in api.list_repo_files(repo, repo_type="dataset")
             if f.startswith("done/") and f.endswith("umap.parquet")]
    print(f"pulling {len(files)} umap parquets from {repo}")
    dest.mkdir(parents=True, exist_ok=True)
    for f in tqdm(files, desc="pull"):
        hf_hub_download(repo, f, repo_type="dataset",
                        local_dir=str(dest), token=token)
    return dest / "done"


def load_catalog(n_use: int) -> dict[str, np.ndarray]:
    """Stream the source dataset once to recover physics labels."""
    from datasets import load_dataset

    from pu.pu_datasets.cosmosweb import CATALOG_COLUMNS
    dataset = os.environ.get(
        "PU_REGRESS_DATASET",
        "<anon>/cosmosweb-hsc-jwst-high-snr-pil2",
    )
    print(f"streaming {dataset} for catalog labels (N={n_use})")
    cols = list(CATALOG_COLUMNS.values())
    raw: dict[str, list] = {c: [] for c in cols}
    ds = load_dataset(dataset, split="train", streaming=True)
    for i, row in enumerate(tqdm(ds, total=n_use, desc="catalog")):
        for c in cols:
            raw[c].append(row[c])
        if i + 1 >= n_use:
            break
    out = {p: np.asarray(raw[col], dtype=np.float32)
           for p, col in CATALOG_COLUMNS.items()}
    out["g-r"] = out["mag_g"] - out["mag_r"]
    return out


def colour_range(name: str, values: np.ndarray) -> tuple[float, float]:
    """Per-property colourmap range. Hard-coded for the redshift physical
    range; everything else uses 1–99 percent quantiles to suppress
    outliers without losing the bulk of the distribution."""
    finite = values[np.isfinite(values)]
    if name == "redshift":
        return 0.0, min(4.0, float(np.quantile(finite, 0.99)))
    lo, hi = np.quantile(finite, [0.01, 0.99])
    return float(lo), float(hi)


def _load_modality_coords(modality: str, done_dir: Path
                          ) -> dict[tuple[str, str], np.ndarray]:
    coords: dict[tuple[str, str], np.ndarray] = {}
    for alias, sizes in FAMILY_SIZES.items():
        for size in sizes:
            p = done_dir / f"{modality}__{alias}_{size}" / "umap.parquet"
            if not p.exists():
                continue
            df = pl.read_parquet(p)
            coords[(alias, size)] = np.stack(
                [df["umap_x"].to_numpy(), df["umap_y"].to_numpy()], axis=1)
    return coords


def render_individual_pngs(modality: str,
                           coords: dict[tuple[str, str], np.ndarray],
                           catalog: dict[str, np.ndarray],
                           keep: np.ndarray, out_dir: Path,
                           dpi: int) -> int:
    """One PNG per (tuple, property)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for prop in PROPERTIES:
        y = catalog[prop]
        vmin, vmax = colour_range(prop, y[keep])
        for (alias, size), xy in coords.items():
            n = min(len(xy), len(y))
            sub = keep[keep < n]
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            sc = ax.scatter(
                xy[sub, 0], xy[sub, 1],
                c=y[sub], cmap="viridis",
                vmin=vmin, vmax=vmax,
                s=3, alpha=0.6, linewidths=0,
            )
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
            ax.set_title(f"{modality} · {alias} · {size}", fontsize=10)
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label(prop, fontsize=9)
            fig.tight_layout()
            png = out_dir / f"{alias}_{size}__{prop}.png"
            fig.savefig(png, dpi=dpi)
            plt.close(fig)
            n_written += 1
    return n_written


def render_modality_pdf(modality: str,
                        coords: dict[tuple[str, str], np.ndarray],
                        catalog: dict[str, np.ndarray],
                        keep: np.ndarray, out_pdf: Path) -> None:
    families = list(FAMILY_SIZES.items())
    n_rows = len(families)
    n_cols = max(len(s) for _, s in families)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        for prop in PROPERTIES:
            y = catalog[prop]
            vmin, vmax = colour_range(prop, y[keep])
            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(2.8 * n_cols, 2.8 * n_rows),
                squeeze=False,
            )
            for r, (alias, sizes) in enumerate(families):
                for c in range(n_cols):
                    ax = axes[r, c]
                    if c >= len(sizes) or (alias, sizes[c]) not in coords:
                        ax.axis("off")
                        continue
                    size = sizes[c]
                    xy = coords[(alias, size)]
                    n = min(len(xy), len(y))
                    sub = keep[keep < n]
                    sc = ax.scatter(
                        xy[sub, 0], xy[sub, 1],
                        c=y[sub], cmap="viridis",
                        vmin=vmin, vmax=vmax,
                        s=2, alpha=0.55, linewidths=0,
                    )
                    ax.set_xticks([]); ax.set_yticks([])
                    if r == 0:
                        ax.set_title(size, fontsize=9)
                    if c == 0:
                        ax.set_ylabel(alias, fontsize=9, rotation=0,
                                      ha="right", va="center", labelpad=20)
            cax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
            fig.colorbar(sc, cax=cax, label=prop)
            fig.suptitle(
                f"UMAP of {modality.upper()} embeddings — coloured by {prop}",
                fontsize=12, y=0.995,
            )
            fig.subplots_adjust(left=0.05, right=0.9, top=0.94, bottom=0.03,
                                wspace=0.05, hspace=0.10)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
    print(f"wrote {out_pdf}")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir or (args.out_dir / "_regress_cache")

    if args.done_dir:
        done_dir = args.done_dir
    elif args.pull_from:
        done_dir = fetch_done(args.pull_from, cache, os.environ.get("HF_TOKEN"))
    else:
        print("[fatal] need --pull-from or --done-dir", file=sys.stderr)
        return 2

    # Recover N from the first available umap.parquet.
    n_use = None
    for tag_dir in sorted(done_dir.iterdir()):
        u = tag_dir / "umap.parquet"
        if u.exists():
            n_use = len(pl.read_parquet(u))
            break
    if n_use is None:
        print(f"[fatal] no umap.parquet under {done_dir}", file=sys.stderr)
        return 1
    catalog = load_catalog(n_use)

    rng = np.random.default_rng(0)
    for modality in MODALITIES:
        coords = _load_modality_coords(modality, done_dir)
        if not coords:
            print(f"  [skip] {modality}: no umap parquets found")
            continue
        n_galaxies = max(len(v) for v in coords.values())
        keep = np.sort(rng.choice(n_galaxies,
                                  size=min(args.n_points, n_galaxies),
                                  replace=False))
        if not args.no_png:
            png_dir = args.out_dir / "umap" / modality
            n = render_individual_pngs(
                modality, coords, catalog, keep, png_dir, args.png_dpi)
            print(f"wrote {n} pngs under {png_dir}/")
        if args.pdf:
            render_modality_pdf(
                modality, coords, catalog, keep,
                args.out_dir / f"umap_{modality}.pdf")

    return 0


if __name__ == "__main__":
    sys.exit(main())
