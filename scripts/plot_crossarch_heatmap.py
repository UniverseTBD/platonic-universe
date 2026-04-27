#!/usr/bin/env python3
"""Heatmap of pairwise calibrated MKNN/CKA across architectures.

Reads cached pairwise matrices from
``data/crossarch_calibrate_{hsc,jwst}.parquet`` and renders a single
figure with two panels (MKNN, CKA). Within each panel the matrix is a
single image where the upper triangle (i<j) holds HSC values and the
lower triangle (i>j) holds JWST values; the diagonal is masked. Both
triangles share one colour scale per metric. Family blocks are
separated by thin white gaps.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIGS_DIR = ROOT / "figs"

MODALITIES = ("hsc", "jwst")
METRICS = ("mknn", "cka")
METRIC_LABEL = {"mknn": "Calibrated MKNN", "cka": "Calibrated CKA"}

# Mirror of plot_crossarchitectural.MODELS. Order is family-first
# then by size. Models not present in the cached parquets are dropped at
# load time, so this list is a superset target.
MODELS: list[tuple[str, str]] = [
    ("vit",       "base"),
    ("vit",       "large"),
    ("vit",       "huge"),
    ("vit-mae",   "base"),
    ("vit-mae",   "large"),
    ("vit-mae",   "huge"),
    ("clip",      "base"),
    ("clip",      "large"),
    ("convnext",  "nano"),
    ("convnext",  "tiny"),
    ("convnext",  "base"),
    ("convnext",  "large"),
    ("dinov3",    "vits16"),
    ("dinov3",    "vits16plus"),
    ("dinov3",    "vitb16"),
    ("dinov3",    "vitl16"),
    ("dinov3",    "vith16plus"),
    ("dinov3",    "vit7b16"),
    ("vjepa",     "large"),
    ("vjepa",     "huge"),
    ("vjepa",     "giant"),
    ("ijepa",     "huge"),
    ("ijepa",     "giant"),
    ("astropt",   "15m"),
    ("astropt",   "95m"),
    ("astropt",   "850m"),
    ("paligemma", "3b"),
    ("paligemma", "10b"),
    ("paligemma", "28b"),
]

FAMILY_GAP = 0.35  # cell-units of white space between adjacent families


def cache_path(data_dir: Path, modality: str) -> Path:
    return data_dir / f"crossarch_calibrate_{modality}.parquet"


def filter_to_available(
    models: list[tuple[str, str]], data_dir: Path
) -> list[tuple[str, str]]:
    """Drop (family, size) entries missing from any modality's parquet."""
    available: set[str] | None = None
    for modality in MODALITIES:
        df = pl.read_parquet(cache_path(data_dir, modality))
        seen = set(df["model_i"].to_list()) | set(df["model_j"].to_list())
        available = seen if available is None else available & seen
    keep: list[tuple[str, str]] = []
    skipped: list[str] = []
    for fam, sz in models:
        label = f"{fam}_{sz}"
        if available is not None and label in available:
            keep.append((fam, sz))
        else:
            skipped.append(label)
    if skipped:
        print(f"Skipping models not in cache: {skipped}")
    return keep


def load_matrix(path: Path, metric: str, labels: list[str]) -> np.ndarray:
    df = pl.read_parquet(path).filter(pl.col("metric") == metric)
    idx = {lab: i for i, lab in enumerate(labels)}
    N = len(labels)
    M = np.full((N, N), np.nan, dtype=np.float64)
    for row in df.iter_rows(named=True):
        i = idx.get(row["model_i"])
        j = idx.get(row["model_j"])
        if i is None or j is None:
            continue
        M[i, j] = row["value"]
    return M


def merge_triangles(hsc: np.ndarray, jwst: np.ndarray) -> np.ndarray:
    """Upper triangle = HSC, lower triangle = JWST, diagonal = NaN."""
    N = hsc.shape[0]
    M = np.full_like(hsc, np.nan)
    iu = np.triu_indices(N, k=1)
    il = np.tril_indices(N, k=-1)
    M[iu] = hsc[iu]
    M[il] = jwst[il]
    return M


def family_boundaries(families: list[str]) -> list[int]:
    return [
        i for i in range(1, len(families))
        if families[i] != families[i - 1]
    ]


def draw_family_gaps(ax, families: list[str], gap: float) -> None:
    """Overlay thin white bands at family boundaries on both axes."""
    N = len(families)
    half = gap / 2
    for b in family_boundaries(families):
        ax.add_patch(Rectangle(
            (-0.5, b - 0.5 - half), N, gap,
            facecolor="white", edgecolor="none", zorder=5,
        ))
        ax.add_patch(Rectangle(
            (b - 0.5 - half, -0.5), gap, N,
            facecolor="white", edgecolor="none", zorder=5,
        ))


def draw_panel(
    fig,
    ax,
    M: np.ndarray,
    labels: list[str],
    families: list[str],
    metric: str,
    cmap_name: str,
) -> None:
    N = len(labels)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("white")

    finite = M[np.isfinite(M)]
    vmin, vmax = float(finite.min()), float(finite.max())

    M_masked = np.ma.masked_invalid(M)
    im = ax.imshow(
        M_masked,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )

    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.tick_params(length=0)

    ax.plot(
        [-0.5, N - 0.5],
        [-0.5, N - 0.5],
        color="black",
        lw=0.5,
        zorder=2,
    )

    draw_family_gaps(ax, families, FAMILY_GAP)

    halo = [pe.withStroke(linewidth=2.2, foreground="black")]
    ax.text(
        N - 1, 0, "HSC",
        ha="right", va="top",
        fontsize=11, fontweight="bold", color="white",
        path_effects=halo, zorder=10,
    )
    ax.text(
        0, N - 1, "JWST",
        ha="left", va="bottom",
        fontsize=11, fontweight="bold", color="white",
        path_effects=halo, zorder=10,
    )

    divider = make_axes_locatable(ax)
    cb_ax = divider.append_axes("right", size="3%", pad=0.08)
    cb = fig.colorbar(im, cax=cb_ax)
    cb.set_label(METRIC_LABEL[metric], fontsize=10)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--out", type=Path, default=FIGS_DIR / "crossarch_heatmap.pdf"
    )
    parser.add_argument("--cmap", default="viridis")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    models = filter_to_available(MODELS, args.data_dir)
    if not models:
        raise RuntimeError("No models from MODELS are present in the cache.")

    labels = [f"{f}_{s}" for f, s in models]
    families = [f for f, _ in models]

    merged: dict[str, np.ndarray] = {}
    for metric in METRICS:
        H = load_matrix(cache_path(args.data_dir, "hsc"), metric, labels)
        J = load_matrix(cache_path(args.data_dir, "jwst"), metric, labels)
        merged[metric] = merge_triangles(H, J)

    fig, axes = plt.subplots(1, len(METRICS), figsize=(11, 5.4))

    for ax, metric in zip(axes, METRICS):
        draw_panel(fig, ax, merged[metric], labels, families, metric, args.cmap)

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.35)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
