#!/usr/bin/env python3
"""
Plot per-model crossmodal Procrustes distance (HSC ↔ JWST) against the
HSC mean physics R² of each model.

For each model the pickle stores a single HSC↔JWST Procrustes distance
per property under ``cross_modal_hsc_vs_jwst[property]``; we average
that over (redshift, mass, sSFR) and plot it against each model's HSC
mean R² across the same three properties (matching the convention in
``plot_crossmodal.py``).

Procrustes distances are read from
``data/procrustes_distances_45000galaxies_upsampled.pkl``. Each entry in
``cross_modal_hsc_vs_jwst[property]`` is a tuple
``(family, raw_size, params, distance)``.

Single panel: one point per model.

Complements ``plot_crossarchitectural_procrustes.py`` (cross-arch
divergence vs R²) and ``plot_crossmodal.py`` (crossmodal MKNN / CKA vs
R²).
"""

import argparse
import json
import pickle
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
from plot_crossmodal import plot_partial  # noqa: E402
from plot_intramodal import FAMILY_STYLE  # noqa: E402
from plot_r2_vs_params import PARAM_COUNTS  # noqa: E402

DEFAULT_PROCRUSTES_PKL = (
    ROOT / "data" / "procrustes_distances_45000galaxies_upsampled.pkl"
)
DEFAULT_R2_JSON = ROOT / "r2_vs_params_45000galaxies_upsampled.json"
R2_PROPS = ("redshift", "mass", "sSFR")

R2_MODALITY = "hsc"


def normalize_size(family: str, size: str) -> str:
    """Map procrustes-pickle size labels to PARAM_COUNTS / R² JSON keys."""
    if family == "astropt":
        # "015M" -> "15m", "095M" -> "95m", "850M" -> "850m"
        s = size.lower().rstrip("m").lstrip("0") or "0"
        return s + "m"
    return size


def load_procrustes(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_r2(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def crossmodal_distances(procrustes: dict) -> dict[tuple[str, str], float]:
    """Mean HSC↔JWST distance per (family, raw_size), averaged over R2_PROPS."""
    base = procrustes["cross_modal_hsc_vs_jwst"][R2_PROPS[0]]
    keys = [(fam, sz) for fam, sz, *_ in base]
    by_key: dict[tuple[str, str], list[float]] = {k: [] for k in keys}
    for prop in R2_PROPS:
        for fam, sz, _params, dist in procrustes["cross_modal_hsc_vs_jwst"][prop]:
            by_key[(fam, sz)].append(float(dist))
    return {k: float(np.nanmean(v)) for k, v in by_key.items()}


def collect_points(
    procrustes: dict,
    r2: dict,
    excluded: set[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (hsc_mean_r2, mean_crossmodal_procrustes, families).

    Distance is the HSC↔JWST Procrustes distance averaged over
    (redshift, mass, sSFR). R² is the HSC mean across the same three
    properties.
    """
    base_order = procrustes["cross_modal_hsc_vs_jwst"][R2_PROPS[0]]
    dist_mean = crossmodal_distances(procrustes)

    r2s, dists, fams = [], [], []
    for family, raw_size, _params, _dist in base_order:
        if family in excluded or family not in FAMILY_STYLE:
            continue
        size = normalize_size(family, raw_size)
        if size not in PARAM_COUNTS.get(family, {}):
            continue
        try:
            r2_val = float(np.mean([
                r2[R2_MODALITY][family][size][p]["r2_mean"] for p in R2_PROPS
            ]))
        except KeyError:
            continue
        r2s.append(r2_val)
        dists.append(dist_mean[(family, raw_size)])
        fams.append(family)

    return np.array(r2s), np.array(dists), fams


def plot_scatter(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    families: list[str],
    xlabel: str,
    ylabel: str,
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
        )

        xlim = ax.get_xlim()
        m, b = np.polyfit(x[finite], y[finite], 1)
        xfit = np.linspace(xlim[0], xlim[1], 200)
        ax.plot(xfit, m * xfit + b, color="gray", lw=2, ls="--", zorder=0)
        ax.set_xlim(xlim)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)


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

    fig, ax = plt.subplots(1, 1, figsize=(3.0, 2.2))

    r2_vals, dists, fams = collect_points(procrustes, r2, exclude_families)
    plot_scatter(
        ax, r2_vals, dists, fams,
        xlabel="HSC [Mean $R^2$]",
        ylabel="$d_{proc}$",
    )
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", fontsize=9, ncol=len(labels) // 2,
        columnspacing=0.55,
        bbox_to_anchor=(0.52, 1.18),
        handletextpad=0.1,
        frameon=False,
    )

    fig.tight_layout()
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

    Fits ``d_proc ~ R² + C(family)`` via OLS and draws the FWL
    added-variable plot (R² and d_proc both demeaned within family),
    annotated with the partial slope β and its Type-II ANOVA p-value.
    """
    fig, ax = plt.subplots(1, 1, figsize=(3.0, 2.2))
    r2_vals, dists, fams = collect_points(procrustes, r2, exclude_families)

    ancova = plot_partial(
        ax, r2_vals, dists, fams,
        xlabel=r"HSC [Mean $\Delta R^2$]",
        ylabel=r"$\Delta d_{proc}$",
    )
    if ancova is not None:
        print(
            f"  ANCOVA  d_proc ~ R² + C(family)  "
            f"slope={ancova['slope']:+.4f}  "
            f"F({ancova['df_num']},{ancova['df_denom']})={ancova['F']:.2f}  "
            f"p={ancova['p']:.3g}  "
            f"(n={ancova['n']}, families={ancova['k']})"
        )
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", fontsize=9, ncol=len(labels) // 2,
        columnspacing=0.55,
        bbox_to_anchor=(0.52, 1.18),
        handletextpad=0.1,
        frameon=False,
    )

    fig.tight_layout()
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
    parser.add_argument("--out-name", default="crossmodal_procrustes.pdf",
                        help="Output filename inside figs/")
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
