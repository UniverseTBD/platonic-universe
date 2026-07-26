#!/usr/bin/env python3
"""
Intramodal MKNN / CKA vs per-property HSC R² — 3×3 appendix figure.

Rows: physics property (redshift, mass, sSFR).
Cols: modality (JWST, Legacy Survey, HSC).
One point per (small, large) pair; x is the pair's intramodal MKNN / CKA
from the manuscript table, y is R² of the property for the larger model
under HSC (from r2_vs_params_45000galaxies_upsampled.json).

Same model/pair set as plot_intramodal.py; this view just breaks
the y-axis back into its three property components instead of averaging.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
FIGS_DIR = ROOT / "figs"

sys.path.insert(0, str(SCRIPTS_DIR))
from plot_intramodal import (  # noqa: E402
    CKA_COL,
    DEFAULT_R2_JSON,
    FAMILY_STYLE,
    MKNN_COL,
    MODALITIES,
    MODALITY_LABEL,
    PAIRS,
    R2_PROPS,
    load_r2_json,
)

PROP_LABEL = {
    "redshift": "redshift",
    "mass": r"$\log M\star$",
    "sSFR": r"$\log\,\mathrm{sSFR}$",
}


def r2_for_larger_prop(r2: dict, family: str, size: str, prop: str) -> float:
    return float(r2["hsc"][family][size][prop]["r2_mean"])


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
            0.97, 0.03,
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
    ax.tick_params(axis="y", direction="in")


def _make_figure(
    pairs: list[tuple],
    r2: dict,
    exclude_families: set[str],
    modalities: tuple[str, ...],
    col_map: dict[str, int],
    metric_label: str,
    out_name: str,
) -> None:
    unknown = exclude_families - set(FAMILY_STYLE)
    if unknown:
        raise ValueError(
            f"Unknown family names in --exclude-families: {sorted(unknown)}. "
            f"Valid: {sorted(FAMILY_STYLE)}"
        )

    kept = [p for p in pairs if p[0] not in exclude_families]
    if not kept:
        raise RuntimeError("No pairs left after applying --exclude-families")

    n_rows = len(R2_PROPS)
    n_cols = len(modalities)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(8, 6),
        sharey="row", sharex="col", squeeze=False,
    )

    for i, prop in enumerate(R2_PROPS):
        for j, modality in enumerate(modalities):
            ax = axes[i, j]
            col = col_map[modality]
            xs, ys, fams = [], [], []
            for p in kept:
                family, _, large = p[0], p[1], p[2]
                xs.append(float(p[col]) / 100.0)
                ys.append(r2_for_larger_prop(r2, family, large, prop))
                fams.append(family)
            plot_panel(ax, np.array(xs), np.array(ys), fams)

            #if i == 0:
            #    ax.set_title(MODALITY_LABEL[modality], fontsize=10)
            if i == n_rows - 1:
                ax.set_xlabel(
                    f"{MODALITY_LABEL[modality]} [{metric_label} %]",
                    fontsize=10,
                )
            if j == 0:
                ax.set_ylabel(
                    rf"{PROP_LABEL[prop]} $[R^2]$ ", fontsize=10,
                )

    seen: dict[str, object] = {}
    for row in axes:
        for ax in row:
            for h, lab in zip(*ax.get_legend_handles_labels()):
                if lab not in seen:
                    seen[lab] = h
    fig.legend(
        seen.values(), list(seen.keys()),
        loc="upper center", fontsize=9, ncol=len(seen),
        columnspacing=0.55,
        bbox_to_anchor=(0.5, 1.0),
        handletextpad=0.1,
        frameon=False,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plt.subplots_adjust(wspace=0.08, hspace=0.08)
    out = FIGS_DIR / out_name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2-json", type=Path, default=DEFAULT_R2_JSON)
    parser.add_argument("--modalities", nargs="+", default=list(MODALITIES),
                        choices=MODALITIES)
    parser.add_argument("--exclude-families", nargs="+", default=["dinov3"],
                        metavar="FAMILY",
                        help=f"Family names to drop (valid: "
                             f"{sorted(FAMILY_STYLE)})")
    args = parser.parse_args()

    FIGS_DIR.mkdir(exist_ok=True)
    r2 = load_r2_json(args.r2_json)
    exclude = set(args.exclude_families)
    modalities = tuple(args.modalities)

    print(f"Loaded R² JSON from {args.r2_json}")
    print(f"Grid: {len(R2_PROPS)}×{len(modalities)} (properties × modalities)")
    print(f"Pairs: {len(PAIRS)} total, "
          f"{len([p for p in PAIRS if p[0] not in exclude])} "
          f"after --exclude-families={args.exclude_families or '[]'}")

    _make_figure(
        PAIRS, r2, exclude, modalities,
        col_map=MKNN_COL, metric_label="MKNN",
        out_name="intramodal_per_property.pdf",
    )
    _make_figure(
        PAIRS, r2, exclude, modalities,
        col_map=CKA_COL, metric_label="CKA",
        out_name="intramodal_cka_per_property.pdf",
    )


if __name__ == "__main__":
    main()
