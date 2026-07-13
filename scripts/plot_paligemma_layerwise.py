#!/usr/bin/env python3
"""Layerwise physics-probe figure for PaliGemma 2 (3B / 10B / 28B).

Reads ``data/paligemma_{3b,10b,28b}_layerwise_probe.parquet`` (written by
``experiments/platonic/regress/04_layerwise_probe.py``) and renders
``figs/paligemma_layerwise_physics.pdf``: R² vs layer depth across the full
PaliGemma stack (SigLIP vision tower → Gemma-2 decoder), as a 3-row grid —
one row per model size (3B, 10B, 28B) — with one column per modality
(HSC, JWST) and one line per property (z, log M*, sSFR).

The paper probes only the final masked-mean hidden state; this figure shows the
whole depth, marks that published layer, and draws the Table-4 R² as a dotted
reference for each property and size.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "figs"

# Model-agnostic via env:
#   PU_PLOT_OUT  output pdf
OUT_PATH = Path(os.environ.get("PU_PLOT_OUT", FIGS / "paligemma_layerwise_physics.pdf"))

# One row per model size, top -> bottom.
SIZES = [
    ("3b",  "PaliGemma 2 (3B)",  ROOT / "data" / "paligemma_3b_layerwise_probe.parquet"),
    ("10b", "PaliGemma 2 (10B)", ROOT / "data" / "paligemma_10b_layerwise_probe.parquet"),
    ("28b", "PaliGemma 2 (28B)", ROOT / "data" / "paligemma_28b_layerwise_probe.parquet"),
]

# Okabe-Ito colourblind-safe hues, one per property (fixed order, not cycled).
PROP_STYLE = {
    "redshift": {"label": r"$z$",             "color": "#0072B2"},
    "mass":     {"label": r"$\log M_\star$",  "color": "#D55E00"},
    "sSFR":     {"label": "sSFR",             "color": "#009E73"},
}

# Paper Table 4 single-layer (published embedding) R², per model size.
# Only the 3B row is published in the paper; other sizes draw no dotted refs.
PAPER_R2_ALL = {
    "3b": {
        "hsc":  {"redshift": 0.458, "mass": 0.539, "sSFR": 0.529},
        "jwst": {"redshift": 0.528, "mass": 0.764, "sSFR": 0.378},
    },
    "10b": {
        "hsc":  {"redshift": 0.464, "mass": 0.537, "sSFR": 0.522},
        "jwst": {"redshift": 0.531, "mass": 0.773, "sSFR": 0.359},
    },
    "28b": {
        "hsc":  {"redshift": 0.398, "mass": 0.492, "sSFR": 0.478},
        "jwst": {"redshift": 0.465, "mass": 0.722, "sSFR": 0.310},
    },
}
PAPER_R2 = PAPER_R2_ALL  # keyed by size in the loop below
MODALITY_TITLE = {"hsc": "HSC", "jwst": "JWST"}


def _draw_panel(ax, df, modality, paper_r2):
    sub = df.filter(pl.col("modality") == modality)
    # Contiguous depth from the sorted unique layer orders (vision <1000,
    # llm 1000-1999, final 2000), robust to layer-count drift.
    orders = sorted(sub["order"].unique().to_list())
    depth_of = {o: i for i, o in enumerate(orders)}
    n_vision = sum(o < 1000 for o in orders)
    n_total = len(orders)

    # SigLIP/Gemma delimiter line.
    if 0 < n_vision < n_total:
        ax.axvline(n_vision - 0.5, ls="--", color="0.45", lw=1, zorder=1)

    for prop, st in PROP_STYLE.items():
        p = sub.filter(pl.col("property") == prop).sort("order")
        if p.is_empty():
            continue
        x = np.array([depth_of[o] for o in p["order"].to_list()])
        y = p["r2_mean"].to_numpy()
        e = p["r2_std"].to_numpy()
        ax.plot(x, y, color=st["color"], lw=1, marker="o", ms=1.5, zorder=3)
        ax.fill_between(x, y - e, y + e, color=st["color"], alpha=0.18,
                        lw=0, zorder=2)


def plot():
    dfs = {key: pl.read_parquet(path) for key, _, path in SIZES if Path(path).exists()}
    sizes = [(k, lbl) for k, lbl, path in SIZES if k in dfs]
    # Rows = modalities (JWST on top, HSC below), columns = model sizes.
    modalities = [m for m in ("jwst", "hsc")
                  if any(m in dfs[k]["modality"].unique().to_list() for k, _ in sizes)]

    n_rows, n_cols = len(modalities), len(sizes)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(8, 3.0),
                             sharex="col", sharey="row", squeeze=False)

    for r, modality in enumerate(modalities):
        for c, (key, label) in enumerate(sizes):
            ax = axes[r, c]
            _draw_panel(ax, dfs[key], modality, PAPER_R2.get(key))
            ax.tick_params(axis="x", direction="in")
            ax.tick_params(axis="y", direction="in")
            if c == 0:
                ax.set_ylabel(f"{MODALITY_TITLE[modality]} [$R^2$]")
            # Column header on the top row only.
            if r != n_rows - 1:
                ax.set_xlabel("")

    # Legend on the top-left panel: property hues + linestyle key.
    prop_handles = [Line2D([0], [0], color=st["color"], lw=2, marker="o", ms=4,
                           label=st["label"]) for st in PROP_STYLE.values()]
    axes[1, 2].legend(handles=prop_handles, fontsize=6,
                      loc="lower right", framealpha=0)

    axes[1, 0].set_xlabel("PaliGemma 3B\n layer depth")
    axes[1, 1].set_xlabel("PaliGemma 10B\n layer depth")
    axes[1, 2].set_xlabel("PaliGemma 28B\n layer depth")

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    plot()
