#!/usr/bin/env python3
"""Layerwise physics-probe figure for PaliGemma 2 (3B).

Reads ``data/paligemma_3b_layerwise_probe.parquet`` (written by
``experiments/platonic/regress/04_layerwise_probe.py``) and renders
``figs/paligemma_layerwise_physics.pdf``: R² vs layer depth across the full
PaliGemma stack (SigLIP vision tower → Gemma-2 decoder), one panel per
modality (HSC, JWST), one line per property (z, log M*, sSFR).

The paper probes only the final masked-mean hidden state; this figure shows the
whole depth, marks that published layer, and draws the Table-4 3B R² as a dotted
reference for each property.
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

# Model-agnostic via env (defaults reproduce the 3B figure):
#   PU_PLOT_INPUT  probe parquet (default the 3B one; PU_LAYERWISE_OUTPUT also honoured)
#   PU_PLOT_MODEL  title label,  e.g. "PaliGemma 2 (10B)"
#   PU_PLOT_OUT    output pdf
#   PU_PLOT_PAPER  key into PAPER_R2_ALL for the dotted Table-4 refs ("" = none)
DATA = Path(os.environ.get("PU_PLOT_INPUT",
            os.environ.get("PU_LAYERWISE_OUTPUT",
                           ROOT / "data" / "paligemma_3b_layerwise_probe.parquet")))
TITLE_MODEL = os.environ.get("PU_PLOT_MODEL", "PaliGemma 2 (3B)")
OUT_PATH = Path(os.environ.get("PU_PLOT_OUT", FIGS / "paligemma_layerwise_physics.pdf"))
PAPER_KEY = os.environ.get("PU_PLOT_PAPER", "3b")

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
PAPER_R2 = PAPER_R2_ALL.get(PAPER_KEY)  # None -> skip dotted refs
MODALITY_TITLE = {"hsc": "HSC", "jwst": "JWST"}


def plot():
    df = pl.read_parquet(DATA)
    modalities = [m for m in ("hsc", "jwst") if m in df["modality"].unique().to_list()]

    fig, axes = plt.subplots(1, len(modalities), figsize=(6 * len(modalities), 5),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, modality in zip(axes, modalities):
        sub = df.filter(pl.col("modality") == modality)
        # Contiguous depth from the sorted unique layer orders (vision <1000,
        # llm 1000-1999, final 2000), robust to layer-count drift.
        orders = sorted(sub["order"].unique().to_list())
        depth_of = {o: i for i, o in enumerate(orders)}
        n_vision = sum(o < 1000 for o in orders)
        n_total = len(orders)

        # SigLIP region shading + vision->LLM boundary.
        if 0 < n_vision < n_total:
            ax.axvspan(-0.5, n_vision - 0.5, color="0.92", zorder=0)
            ax.axvline(n_vision - 0.5, ls="--", color="0.45", lw=1, zorder=1)

        for prop, st in PROP_STYLE.items():
            p = sub.filter(pl.col("property") == prop).sort("order")
            if p.is_empty():
                continue
            x = np.array([depth_of[o] for o in p["order"].to_list()])
            y = p["r2_mean"].to_numpy()
            e = p["r2_std"].to_numpy()
            ax.plot(x, y, color=st["color"], lw=2, marker="o", ms=3, zorder=3)
            ax.fill_between(x, y - e, y + e, color=st["color"], alpha=0.18,
                            lw=0, zorder=2)
            if PAPER_R2 is not None:
                ax.axhline(PAPER_R2[modality][prop], color=st["color"], ls=":",
                           lw=1, alpha=0.6, zorder=1)
            # Star the published-embedding point.
            pub = p.filter(pl.col("is_published"))
            if not pub.is_empty():
                px = depth_of[pub["order"][0]]
                ax.plot(px, pub["r2_mean"][0], marker="*", ms=13,
                        color=st["color"], mec="white", mew=0.6, zorder=4)

        # Region labels.
        if 0 < n_vision < n_total:
            ax.text((n_vision - 1) / 2, 0.98, "SigLIP\nvision tower",
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=8, color="0.35")
            ax.text((n_vision + n_total - 1) / 2, 0.98, "Gemma-2\ndecoder",
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=8, color="0.35")

        ax.set_title(MODALITY_TITLE[modality])
        ax.set_xlabel("Layer depth  (vision → language)")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(r"$R^2$  (linear probe)")

    # Legend: property hues + linestyle key.
    prop_handles = [Line2D([0], [0], color=st["color"], lw=2, marker="o", ms=4,
                           label=st["label"]) for st in PROP_STYLE.values()]
    key_handles = [
        Line2D([0], [0], color="0.4", lw=2, label="this work (per layer)"),
        Line2D([0], [0], color="0.4", marker="*", ms=10, ls="none",
               label="published embedding"),
    ]
    if PAPER_R2 is not None:
        key_handles.insert(1, Line2D([0], [0], color="0.4", ls=":", lw=1,
                                     label=f"paper Table 4 ({PAPER_KEY.upper()})"))
    axes[0].legend(handles=prop_handles + key_handles, fontsize=8,
                   loc="lower right", framealpha=0.9)

    fig.suptitle(f"{TITLE_MODEL}: physics-probe $R^2$ across network depth",
                 fontsize=13)
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    plot()
