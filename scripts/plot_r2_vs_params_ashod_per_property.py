#!/usr/bin/env python3
"""
Parameter count vs per-property R² — appendix figure to
plot_r2_vs_params_ashod.py.

Rows: physics property (redshift, mass, sSFR).
Cols: modality (HSC, JWST).
One point per (family, size); x is the model's parameter count, y is
R² of the property under the column's modality (from Ashod's
``r2_vs_params_45000galaxies_upsampled.json``).
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
from plot_r2_vs_params_ashod import (  # noqa: E402
    DEFAULT_R2_JSON,
    FAMILY_STYLE,
    MODALITIES,
    MODALITY_LABEL,
    PARAM_COUNTS,
    R2_PROPS,
    load_r2_json,
)

PROP_LABEL = {
    "redshift": "redshift",
    "mass": r"$\log M\star$",
    "sSFR": r"$\log\,\mathrm{sSFR}$",
}


def r2_for_model(r2: dict, modality: str, family: str, size: str,
                 prop: str) -> float:
    return float(r2[modality][family][size][prop]["r2_mean"])


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
            s=45, label=style["label"], edgecolors="black", linewidths=0.3,
        )

    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() >= 3:
        rho, p_rho = spearmanr(x[finite], y[finite])
        ax.text(
            0.97, 0.03,
            f"ρ = {rho:.2f}  (p = {p_rho:.1g})",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=8,
        )

    ax.set_xscale("log")
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")


def make_figure(
    r2: dict,
    exclude_families: set[str],
    modalities: tuple[str, ...],
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
        figsize=(2.9 * n_cols + 0.6, 2.4 * n_rows + 0.4),
        sharex="col", squeeze=False,
    )

    for i, prop in enumerate(R2_PROPS):
        for j, modality in enumerate(modalities):
            ax = axes[i, j]
            xs, ys, fams = [], [], []
            for family, sizes in PARAM_COUNTS.items():
                if family in exclude_families:
                    continue
                for size, n_params in sizes.items():
                    try:
                        r2_val = r2_for_model(r2, modality, family, size, prop)
                    except KeyError:
                        continue
                    xs.append(float(n_params))
                    ys.append(r2_val)
                    fams.append(family)
            plot_panel(ax, np.array(xs), np.array(ys), fams)

            if i == n_rows - 1:
                ax.set_xlabel(
                    f"{MODALITY_LABEL[modality]} [Parameters]",
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
        bbox_to_anchor=(0.5, 1.02),
        handletextpad=0.1,
        frameon=False,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plt.subplots_adjust(wspace=0.2, hspace=0.08)
    out = FIGS_DIR / "r2_vs_params_ashod_per_property.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2-json", type=Path, default=DEFAULT_R2_JSON)
    parser.add_argument("--modalities", nargs="+", default=list(MODALITIES),
                        choices=MODALITIES)
    parser.add_argument("--exclude-families", nargs="*", default=[],
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
    n_total = sum(len(v) for v in PARAM_COUNTS.values())
    n_kept = sum(
        len(v) for f, v in PARAM_COUNTS.items() if f not in exclude
    )
    print(f"Models: {n_total} total, "
          f"{n_kept} after --exclude-families={args.exclude_families or '[]'}")

    make_figure(r2, exclude, modalities)


if __name__ == "__main__":
    main()
