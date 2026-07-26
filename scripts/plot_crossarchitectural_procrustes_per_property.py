#!/usr/bin/env python3
"""
Cross-architectural Procrustes distance vs per-property R² — appendix figure.

Rows: physics property (redshift, mass, sSFR).
Cols: modality (HSC, JWST).

For each (modality, property) we reconstruct the M×M Procrustes matrix
from ``cross_model_pairwise[modality][property]`` in
``data/procrustes_distances_45000galaxies_upsampled.pkl``, drop excluded
/ un-styled families, and take each surviving model's mean pairwise
distance to the rest of the cross-family pool. That distance is the
x-axis; the y-axis is the same model's R² for the row's property under
the column's modality.

Per-property counterpart of ``plot_crossarchitectural_procrustes.py``,
which averages over (redshift, mass, sSFR) instead of stratifying.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
FIGS_DIR = ROOT / "figs"

sys.path.insert(0, str(SCRIPTS_DIR))
from plot_crossarchitectural_procrustes import (  # noqa: E402
    DEFAULT_PROCRUSTES_PKL,
    DEFAULT_R2_JSON,
    FAMILY_STYLE,
    MODALITIES,
    MODALITY_LABEL,
    PARAM_COUNTS,
    R2_PROPS,
    load_procrustes,
    load_r2,
    model_order,
    normalize_size,
    reconstruct_pairwise,
)

PROP_LABEL = {
    "redshift": "redshift",
    "mass": r"$\log M\star$",
    "sSFR": r"$\log\,\mathrm{sSFR}$",
}


def collect_points_for(
    procrustes: dict,
    r2: dict,
    modality: str,
    prop: str,
    excluded: set[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (procrustes, r2, families) for one (modality, property)."""
    order = model_order(procrustes, prop)
    n = len(order)
    mat = reconstruct_pairwise(
        procrustes["cross_model_pairwise"][modality][prop], n
    )

    keep_idx, r2s, fams = [], [], []
    for i, (family, raw_size) in enumerate(order):
        if family in excluded or family not in FAMILY_STYLE:
            continue
        size = normalize_size(family, raw_size)
        if size not in PARAM_COUNTS.get(family, {}):
            continue
        try:
            r2_val = float(r2[modality][family][size][prop]["r2_mean"])
        except KeyError:
            continue
        keep_idx.append(i)
        r2s.append(r2_val)
        fams.append(family)

    sub = mat[np.ix_(keep_idx, keep_idx)].copy()
    np.fill_diagonal(sub, np.nan)
    dists = np.nanmean(sub, axis=1)
    return dists, np.array(r2s), fams


def plot_panel(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    families: list[str],
) -> None:
    for family in FAMILY_STYLE:
        mask = np.array([f == family for f in families])
        if not mask.any():
            continue
        style = FAMILY_STYLE[family]
        ax.scatter(
            x[mask], y[mask],
            color=style["color"], marker=style["marker"],
            s=30, label=style["label"], edgecolors="black", linewidths=0.4,
        )

    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() >= 3:
        rho, p_rho = spearmanr(x[finite], y[finite])
        ax.text(
            0.95, 0.75,
            f"ρ = {rho:.3f}  (p = {p_rho:.1g})\n",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
            bbox=None,
        )

        m, b = np.polyfit(x[finite], y[finite], 1)
        xlim = ax.get_xlim()
        xfit = np.linspace(xlim[0], xlim[1], 200)
        ax.plot(xfit, m * xfit + b, color="gray", lw=2, ls="--", zorder=0)
        ax.set_xlim(xlim)

    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="x", direction="in", which="minor")
    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="y", direction="in", which="minor")


def make_figure(
    procrustes: dict,
    r2: dict,
    exclude_families: set[str],
    modalities: tuple[str, ...],
    out_name: str,
) -> None:
    unknown = exclude_families - set(FAMILY_STYLE)
    if unknown:
        raise ValueError(
            f"Unknown family names in --exclude-families: {sorted(unknown)}. "
            f"Valid: {sorted(FAMILY_STYLE)}"
        )

    n_rows = len(R2_PROPS)
    n_cols = len(modalities)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6, 6),
        squeeze=False,
    )

    for i, prop in enumerate(R2_PROPS):
        for j, modality in enumerate(modalities):
            ax = axes[i, j]
            dists, r2_vals, fams = collect_points_for(
                procrustes, r2, modality, prop, exclude_families,
            )
            plot_panel(ax, r2_vals, dists, fams)

            if i == n_rows - 1:
                ax.set_xlabel(
                    rf"{MODALITY_LABEL[modality]} $[R^2]$",
                    fontsize=10,
                )
            if j == 0:
                ax.set_ylabel(
                    rf"{PROP_LABEL[prop]} $[d_{{proc}}]$", fontsize=10,
                )

    seen: dict[str, object] = {}
    for row in axes:
        for ax in row:
            for h, lab in zip(*ax.get_legend_handles_labels()):
                if lab not in seen:
                    seen[lab] = h
    fig.legend(
        seen.values(), list(seen.keys()),
        loc="upper center", fontsize=9, ncol=(len(seen)//2),
        columnspacing=0.55,
        bbox_to_anchor=(0.52, 1.06),
        handletextpad=0.1,
        frameon=False,
    )

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.15)
    out = FIGS_DIR / out_name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--procrustes-pkl", type=Path,
                        default=DEFAULT_PROCRUSTES_PKL,
                        help="Path to procrustes distances pickle")
    parser.add_argument("--r2-json", type=Path, default=DEFAULT_R2_JSON,
                        help="Path to r2_vs_params JSON")
    parser.add_argument("--modalities", nargs="+", default=list(MODALITIES),
                        choices=MODALITIES,
                        help="Subset of modality columns to plot")
    parser.add_argument("--exclude-families", nargs="*", default=["dinov3"],
                        metavar="FAMILY",
                        help=f"Family names to drop (valid: "
                             f"{sorted(FAMILY_STYLE)})")
    parser.add_argument(
        "--out-name",
        default="crossarchitectural_procrustes_per_property.pdf",
        help="Output filename inside figs/",
    )
    args = parser.parse_args()

    FIGS_DIR.mkdir(exist_ok=True)

    procrustes = load_procrustes(args.procrustes_pkl)
    r2 = load_r2(args.r2_json)
    print(f"Loaded procrustes pickle from {args.procrustes_pkl}")
    print(f"Loaded R² JSON from {args.r2_json}")
    print(f"Cols (modalities): {args.modalities}")
    print(f"Excluded families: {args.exclude_families}")

    make_figure(
        procrustes, r2,
        exclude_families=set(args.exclude_families),
        modalities=tuple(args.modalities),
        out_name=args.out_name,
    )


if __name__ == "__main__":
    main()
