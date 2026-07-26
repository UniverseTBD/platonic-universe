#!/usr/bin/env python3
"""
Crossmodal Procrustes distance vs per-property HSC R² — appendix figure.

One panel per physics property (redshift, mass, sSFR).

For each property we read the per-model HSC↔JWST Procrustes distance
from ``cross_modal_hsc_vs_jwst[property]`` in
``data/procrustes_distances_45000galaxies_upsampled.pkl``. That distance
is the y-axis; the x-axis is the same model's HSC R² for the property.

Per-property counterpart of ``plot_crossmodal_procrustes.py``, which
averages over (redshift, mass, sSFR) instead of stratifying.
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
from plot_crossmodal import (  # noqa: E402
    ancova_partial_slope,
    family_demean,
)
from plot_crossmodal_procrustes import (  # noqa: E402
    DEFAULT_PROCRUSTES_PKL,
    DEFAULT_R2_JSON,
    FAMILY_STYLE,
    PARAM_COUNTS,
    R2_MODALITY,
    R2_PROPS,
    load_procrustes,
    load_r2,
    normalize_size,
)

PROP_LABEL = {
    "redshift": "redshift",
    "mass": r"$\log M\star$",
    "sSFR": r"$\log\,\mathrm{sSFR}$",
}


def collect_points_for(
    procrustes: dict,
    r2: dict,
    prop: str,
    excluded: set[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (hsc_r2, procrustes, families) for one property."""
    r2s, dists, fams = [], [], []
    for family, raw_size, _params, dist in procrustes["cross_modal_hsc_vs_jwst"][prop]:
        if family in excluded or family not in FAMILY_STYLE:
            continue
        size = normalize_size(family, raw_size)
        if size not in PARAM_COUNTS.get(family, {}):
            continue
        try:
            r2_val = float(r2[R2_MODALITY][family][size][prop]["r2_mean"])
        except KeyError:
            continue
        r2s.append(r2_val)
        dists.append(float(dist))
        fams.append(family)
    return np.array(r2s), np.array(dists), fams


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


def plot_panel_partial(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    families: list[str],
) -> dict | None:
    """Family-demeaned added-variable panel for ``y ~ x + C(family)``.

    Both axes are within-family residuals so the OLS slope through
    them equals β₁ in the full model (Frisch–Waugh–Lovell). Annotates
    β and the Type-II ANOVA p-value.
    """
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3:
        return None
    fams = [f for f, ok in zip(families, finite) if ok]
    x_f = x[finite]
    y_f = y[finite]
    x_res = family_demean(x_f, fams)
    y_res = family_demean(y_f, fams)

    for family in FAMILY_STYLE:
        mask = np.array([f == family for f in fams])
        if not mask.any():
            continue
        st = FAMILY_STYLE[family]
        ax.scatter(
            x_res[mask], y_res[mask],
            color=st["color"], marker=st["marker"], s=30,
            label=st["label"], edgecolors="black", linewidths=0.4,
        )

    ancova = ancova_partial_slope(x_f, y_f, fams)
    m, b = np.polyfit(x_res, y_res, 1)
    xlim = ax.get_xlim()
    xfit = np.linspace(xlim[0], xlim[1], 200)
    ax.plot(xfit, m * xfit + b, color="gray", lw=2, ls="--", zorder=0)
    ax.set_xlim(xlim)
    ax.text(
        0.05, 0.95,
        f"β = {ancova['slope']:.3f} (p = {ancova['p']:.1g})",
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
    )
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    return ancova


def make_figure(
    procrustes: dict,
    r2: dict,
    exclude_families: set[str],
    out_name: str,
) -> None:
    unknown = exclude_families - set(FAMILY_STYLE)
    if unknown:
        raise ValueError(
            f"Unknown family names in --exclude-families: {sorted(unknown)}. "
            f"Valid: {sorted(FAMILY_STYLE)}"
        )

    n_cols = len(R2_PROPS)
    fig, axes = plt.subplots(1, n_cols, figsize=(6, 2.4), squeeze=False, sharey=True)

    for j, prop in enumerate(R2_PROPS):
        ax = axes[0, j]
        r2_vals, dists, fams = collect_points_for(
            procrustes, r2, prop, exclude_families,
        )
        plot_panel(ax, r2_vals, dists, fams)
        ax.set_xlabel(rf"{PROP_LABEL[prop]} $[R^2]$", fontsize=10)
        if j == 0:
            ax.set_ylabel("$d_{proc}$", fontsize=10)

    seen: dict[str, object] = {}
    for ax in axes.ravel():
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in seen:
                seen[lab] = h
    fig.legend(
        seen.values(), list(seen.keys()),
        loc="upper center", fontsize=9, ncol=(len(seen) // 2 + 1),
        columnspacing=0.4,
        bbox_to_anchor=(0.52, 1.12),
        handletextpad=0.05,
        frameon=False,
    )

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    out = FIGS_DIR / out_name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def make_partial_figure(
    procrustes: dict,
    r2: dict,
    exclude_families: set[str],
    out_name: str,
) -> None:
    """Family-demeaned ANCOVA companion to ``make_figure``.

    Per property fits ``d_proc ~ R² + C(family)`` and draws the FWL
    added-variable panel.
    """
    n_cols = len(R2_PROPS)
    fig, axes = plt.subplots(1, n_cols, figsize=(2.6 * n_cols, 2.4), squeeze=False, sharey=True)

    for j, prop in enumerate(R2_PROPS):
        ax = axes[0, j]
        r2_vals, dists, fams = collect_points_for(
            procrustes, r2, prop, exclude_families,
        )
        ancova = plot_panel_partial(ax, r2_vals, dists, fams)
        if ancova is not None:
            print(
                f"  ANCOVA  d_proc ~ R² + C(family)  prop={prop:<8s}  "
                f"slope={ancova['slope']:+.4f}  "
                f"F({ancova['df_num']},{ancova['df_denom']})={ancova['F']:.2f}  "
                f"p={ancova['p']:.3g}  "
                f"(n={ancova['n']}, families={ancova['k']})"
            )
        ax.set_xlabel(rf"{PROP_LABEL[prop]} $[\Delta R^2]$", fontsize=10)
        if j == 0:
            ax.set_ylabel(r"$\Delta d_{proc}$", fontsize=10, labelpad=-5)

    seen: dict[str, object] = {}
    for ax in axes.ravel():
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in seen:
                seen[lab] = h
    fig.legend(
        seen.values(), list(seen.keys()),
        loc="upper center", fontsize=9, ncol=len(seen),
        columnspacing=0.2,
        bbox_to_anchor=(0.52, 1.06),
        handletextpad=0.05,
        frameon=False,
    )

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05)
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
    parser.add_argument("--exclude-families", nargs="*", default=["dinov3"],
                        metavar="FAMILY",
                        help=f"Family names to drop (valid: "
                             f"{sorted(FAMILY_STYLE)})")
    parser.add_argument(
        "--out-name",
        default="crossmodal_procrustes_per_property.pdf",
        help="Output filename inside figs/",
    )
    args = parser.parse_args()

    FIGS_DIR.mkdir(exist_ok=True)

    procrustes = load_procrustes(args.procrustes_pkl)
    r2 = load_r2(args.r2_json)
    print(f"Loaded procrustes pickle from {args.procrustes_pkl}")
    print(f"Loaded R² JSON from {args.r2_json}")
    print(f"Excluded families: {args.exclude_families}")

    exclude = set(args.exclude_families)
    make_figure(
        procrustes, r2,
        exclude_families=exclude,
        out_name=args.out_name,
    )
    partial_name = args.out_name.replace(".pdf", "_partial.pdf")
    if partial_name == args.out_name:
        partial_name = args.out_name + "_partial"
    make_partial_figure(
        procrustes, r2,
        exclude_families=exclude,
        out_name=partial_name,
    )


if __name__ == "__main__":
    main()
